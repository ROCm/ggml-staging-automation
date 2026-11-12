// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Score a bounded continuation after a long corpus prefix without retaining
// context_length * vocabulary_size logits in memory.

#include "llama.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + path);
    }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("corpus is too large to tokenize");
    }
    const int32_t needed = -llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                           nullptr, 0, true, false);
    if (needed <= 0) {
        throw std::runtime_error("tokenizer did not return a token count");
    }
    std::vector<llama_token> tokens(static_cast<size_t>(needed));
    const int32_t actual = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                          tokens.data(), needed, true, false);
    if (actual != needed) {
        throw std::runtime_error("token count changed while tokenizing corpus");
    }
    return tokens;
}

int32_t read_training_context(const std::string & model_path) {
    gguf_init_params params = {true, nullptr};
    gguf_context * metadata = gguf_init_from_file(model_path.c_str(), params);
    if (!metadata) {
        throw std::runtime_error("could not read GGUF metadata");
    }
    const int64_t architecture_key = gguf_find_key(metadata, "general.architecture");
    if (architecture_key < 0) {
        gguf_free(metadata);
        throw std::runtime_error("GGUF has no general.architecture key");
    }
    const std::string architecture = gguf_get_val_str(metadata, architecture_key);
    const std::string context_key_name = architecture + ".context_length";
    const int64_t context_key = gguf_find_key(metadata, context_key_name.c_str());
    if (context_key < 0) {
        gguf_free(metadata);
        throw std::runtime_error("GGUF has no " + context_key_name + " key");
    }
    const gguf_type type = gguf_get_kv_type(metadata, context_key);
    int32_t context = 0;
    if (type == GGUF_TYPE_UINT32) {
        context = static_cast<int32_t>(gguf_get_val_u32(metadata, context_key));
    } else if (type == GGUF_TYPE_INT32) {
        context = gguf_get_val_i32(metadata, context_key);
    } else {
        gguf_free(metadata);
        throw std::runtime_error(context_key_name + " is not a 32-bit integer");
    }
    gguf_free(metadata);
    if (context <= 0) {
        throw std::runtime_error(context_key_name + " is not positive");
    }
    return context;
}

void inspect(const std::string & model_path) {
    std::cout << "{\"model_context\":" << read_training_context(model_path) << "}\n";
}

double negative_log_likelihood(const float * logits, int32_t vocab_size, llama_token target,
                               int32_t position) {
    if (!logits || target < 0 || target >= vocab_size) {
        throw std::runtime_error("invalid logits or target at token position " + std::to_string(position));
    }
    double maximum = -std::numeric_limits<double>::infinity();
    for (int32_t i = 0; i < vocab_size; ++i) {
        const double value = logits[i];
        if (!std::isfinite(value)) {
            throw std::runtime_error("nonfinite vocabulary logit at token position " + std::to_string(position));
        }
        maximum = std::max(maximum, value);
    }
    double exponential_sum = 0.0;
    for (int32_t i = 0; i < vocab_size; ++i) {
        exponential_sum += std::exp(static_cast<double>(logits[i]) - maximum);
    }
    const double nll = maximum + std::log(exponential_sum) - logits[target];
    if (!std::isfinite(nll)) {
        throw std::runtime_error("nonfinite token likelihood at position " + std::to_string(position));
    }
    return nll;
}

void prefill(llama_context * context, llama_batch & batch, const std::vector<llama_token> & tokens,
             int32_t count, int32_t batch_size) {
    for (int32_t offset = 0; offset < count; offset += batch_size) {
        const int32_t length = std::min(batch_size, count - offset);
        batch.n_tokens = length;
        for (int32_t i = 0; i < length; ++i) {
            const int32_t position = offset + i;
            batch.token[i] = tokens[static_cast<size_t>(position)];
            batch.pos[i] = position;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = position == count - 1 ? 1 : 0;
        }
        if (llama_decode(context, batch) != 0) {
            throw std::runtime_error("prefill failed at token offset " + std::to_string(offset));
        }
    }
}

void run(const std::string & model_path, const std::string & corpus_path,
         const std::string & device_name, int32_t context_length, int32_t score_tokens) {
    if (context_length < 2 || score_tokens < 1 || score_tokens >= context_length) {
        throw std::runtime_error("context length must exceed the positive scored-token count");
    }

    const int32_t training_context = read_training_context(model_path);
    if (context_length > training_context) {
        std::cout << "{\"status\":\"unsupported\",\"reason\":\"requested context exceeds model training context\","
                  << "\"context_length\":" << context_length << ",\"model_context\":" << training_context << "}\n";
        return;
    }

    llama_backend_init();
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name(device_name.c_str());
    if (!device) {
        throw std::runtime_error("backend device is unavailable: " + device_name);
    }
    ggml_backend_dev_t devices[] = {device, nullptr};
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    model_params.devices = devices;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        throw std::runtime_error("model load failed");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = tokenize(vocab, read_file(corpus_path));
    if (tokens.size() < static_cast<size_t>(context_length)) {
        throw std::runtime_error("corpus has fewer tokens than the requested context length");
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(context_length);
    context_params.n_batch = 2048;
    context_params.n_ubatch = 512;
    context_params.n_seq_max = 1;
    context_params.n_outputs_max = 1;
    context_params.n_outputs_max_per_seq = 1;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_params.type_k = GGML_TYPE_F16;
    context_params.type_v = GGML_TYPE_F16;
    context_params.no_perf = true;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) {
        throw std::runtime_error("context allocation failed");
    }
    if (llama_n_ctx(context) != static_cast<uint32_t>(context_length)) {
        throw std::runtime_error("runtime context differs from requested context");
    }

    constexpr int32_t batch_size = 2048;
    const int32_t prefix_tokens = context_length - score_tokens;
    llama_batch batch = llama_batch_init(batch_size, 0, 1);
    prefill(context, batch, tokens, prefix_tokens, batch_size);

    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(context, -1);
    double total_nll = 0.0;
    double max_nll = 0.0;
    for (int32_t i = 0; i < score_tokens; ++i) {
        const int32_t position = prefix_tokens + i;
        const llama_token target = tokens[static_cast<size_t>(position)];
        const double nll = negative_log_likelihood(logits, vocab_size, target, position);
        total_nll += nll;
        max_nll = std::max(max_nll, nll);
        if (i + 1 < score_tokens) {
            batch.n_tokens = 1;
            batch.token[0] = target;
            batch.pos[0] = position;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            if (llama_decode(context, batch) != 0) {
                throw std::runtime_error("teacher-forced decode failed at token position " + std::to_string(position));
            }
            logits = llama_get_logits_ith(context, -1);
        }
    }

    const double mean_nll = total_nll / score_tokens;
    const double perplexity = std::exp(mean_nll);
    if (!std::isfinite(perplexity)) {
        throw std::runtime_error("perplexity is nonfinite");
    }
    std::cout << std::setprecision(17)
              << "{\"status\":\"ok\",\"context_length\":" << context_length
              << ",\"model_context\":" << training_context << ",\"prefix_tokens\":" << prefix_tokens
              << ",\"scored_tokens\":" << score_tokens << ",\"vocab_size\":" << vocab_size
              << ",\"mean_nll\":" << mean_nll << ",\"ppl\":" << perplexity
              << ",\"max_nll\":" << max_nll << "}\n";

    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "--inspect") {
        try {
            inspect(argv[2]);
            return 0;
        } catch (const std::exception & error) {
            std::cerr << "ERROR " << error.what() << '\n';
            return 1;
        }
    }
    if (argc != 6) {
        std::cerr << "usage: hrx-stream-perplexity --inspect MODEL | MODEL CORPUS DEVICE CONTEXT_LENGTH SCORE_TOKENS\n";
        return 2;
    }
    try {
        run(argv[1], argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]));
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "ERROR " << error.what() << '\n';
        return 1;
    }
}
