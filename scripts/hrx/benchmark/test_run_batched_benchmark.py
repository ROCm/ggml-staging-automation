# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Exercise download retries without network requests or real sleeps."""

import hashlib
import io
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import Mock, call, patch

import run_batched_benchmark as benchmark


class DownloadRetryTest(unittest.TestCase):
    def setUp(self):
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        self.models_dir = Path(temporary_dir.name)
        self.payload = b"verified model bytes"
        self.model = benchmark.ModelSpec(
            id="test-model",
            tier="all",
            expected_results={},
            test_modes={},
            name="Test-Model-GGUF",
            repository="test/model",
            revision="a" * 40,
            filename="model.gguf",
            size_bytes=len(self.payload),
            sha256=hashlib.sha256(self.payload).hexdigest(),
        )
        self.opener = Mock()
        sleep_patch = patch.object(benchmark.time, "sleep")
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)
        clock_patch = patch.object(benchmark.time, "time", return_value=1445412420.0)
        clock_patch.start()  # Wed, 21 Oct 2015 07:27:00 GMT
        self.addCleanup(clock_patch.stop)
        log_patch = patch.object(benchmark, "log")
        self.log = log_patch.start()
        self.addCleanup(log_patch.stop)

    def http_error(self, value=None, code=429):
        headers = Message()
        if value is not None:
            headers["retry-after"] = value
        return urllib.error.HTTPError(
            self.model.download_url, code, "test failure", headers, io.BytesIO()
        )

    def response(self):
        response = io.BytesIO(self.payload)
        response.status = 200
        return response

    def download(self, *outcomes, **kwargs):
        self.opener.side_effect = outcomes
        return benchmark.download_model(
            self.model, self.models_dir, urlopen=self.opener, **kwargs
        )

    def test_seconds_header_is_honored_and_download_is_verified(self):
        for code in (429, 503):
            with self.subTest(code=code):
                self.sleep.reset_mock()
                error = self.http_error(" 120 ", code)
                path = self.download(error, self.response())
                self.sleep.assert_called_once_with(120.0)
                self.assertEqual(path.read_bytes(), self.payload)
                self.assertTrue(error.closed)
                self.assertFalse(list(path.parent.glob("*.tmp")))
                self.log.assert_any_call(
                    "Download attempt 1/3 failed for test-model: "
                    f"HTTP Error {code}: test failure; retrying in 120s from byte zero"
                )

    def test_http_date_formats_are_utc(self):
        for value in (
            "Wed, 21 Oct 2015 07:28:00 GMT",
            "Wednesday, 21-Oct-15 07:28:00 GMT",
            "Wed Oct 21 07:28:00 2015",
        ):
            with self.subTest(value=value):
                self.sleep.reset_mock()
                self.download(self.http_error(value), self.response())
                self.sleep.assert_called_once_with(60.0)

    def test_zero_and_past_date_keep_local_backoff(self):
        for value in ("0", "Wed, 21 Oct 2015 07:26:00 GMT"):
            with self.subTest(value=value):
                self.sleep.reset_mock()
                self.download(self.http_error(value), self.response())
                self.sleep.assert_called_once_with(1.0)

    def test_missing_or_malformed_429_header_uses_exponential_backoff(self):
        for value in (None, "", "invalid", "-1", "1.5", "NaN", "inf", "١٢", "9" * 400):
            with self.subTest(value=value):
                self.sleep.reset_mock()
                self.download(
                    self.http_error(value), self.http_error(value), self.response()
                )
                self.assertEqual(self.sleep.call_args_list, [call(60.0), call(120.0)])

    def test_other_failures_keep_existing_backoff(self):
        for error in (OSError("connection reset"), self.http_error(code=503)):
            with self.subTest(error=str(error)):
                self.sleep.reset_mock()
                self.download(error, error, self.response())
                self.assertEqual(self.sleep.call_args_list, [call(1.0), call(2.0)])

    def test_new_header_is_used_on_each_attempt(self):
        self.download(self.http_error("120"), self.http_error("300"), self.response())
        self.assertEqual(self.sleep.call_args_list, [call(120.0), call(300.0)])

    def test_header_is_honored_even_when_local_backoff_is_disabled(self):
        self.download(self.http_error("120"), self.response(), retry_delay=0)
        self.sleep.assert_called_once_with(120.0)

    def test_exhaustion_does_not_sleep_after_last_attempt_or_publish_model(self):
        errors = [self.http_error("120") for _ in range(3)]
        with self.assertRaisesRegex(benchmark.DownloadError, "after 3 attempt.*429"):
            self.download(*errors)
        self.assertEqual(self.opener.call_count, 3)
        self.assertEqual(self.sleep.call_args_list, [call(120.0), call(120.0)])
        self.assertTrue(all(error.closed for error in errors))
        destination = self.models_dir / self.model.name / self.model.filename
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
