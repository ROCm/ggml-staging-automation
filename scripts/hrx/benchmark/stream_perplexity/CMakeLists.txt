cmake_minimum_required(VERSION 3.16)
project(hrx_stream_perplexity LANGUAGES CXX)

find_package(llama CONFIG REQUIRED)

add_executable(hrx-stream-perplexity stream_perplexity.cpp)
target_compile_features(hrx-stream-perplexity PRIVATE cxx_std_17)
target_link_libraries(hrx-stream-perplexity PRIVATE llama)
set_target_properties(hrx-stream-perplexity PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")
install(TARGETS hrx-stream-perplexity RUNTIME DESTINATION bin)
