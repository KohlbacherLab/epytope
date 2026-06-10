from unittest import TestCase
from unittest import mock

import requests

from epytope.IO.EnsemblRESTAdapter import (
    EnsemblRESTAdapter,
    EnsemblRESTError,
    EnsemblRateLimitError,
    EnsemblConnectionError,
    _RateLimiter,
)


class TestEnsemblExceptions(TestCase):
    def test_exception_hierarchy(self):
        self.assertTrue(issubclass(EnsemblRateLimitError, EnsemblRESTError))
        self.assertTrue(issubclass(EnsemblConnectionError, EnsemblRESTError))
        self.assertTrue(issubclass(EnsemblRESTError, Exception))


class TestRateLimiter(TestCase):
    def test_blocks_when_window_exceeded(self):
        clock = {"t": 1000.0}
        sleeps = []

        def fake_sleep(secs):
            sleeps.append(secs)
            clock["t"] += secs

        limiter = _RateLimiter([(2, 10.0)])
        with mock.patch("time.monotonic",
                        lambda: clock["t"]), \
             mock.patch("time.sleep", fake_sleep):
            limiter.acquire()   # 1st call: fits
            limiter.acquire()   # 2nd call: fits
            limiter.acquire()   # 3rd call: window full -> must wait 10s

        self.assertEqual(sleeps, [10.0])

    def test_no_block_under_limit(self):
        clock = {"t": 500.0}
        sleeps = []
        limiter = _RateLimiter([(5, 1.0)])
        with mock.patch("time.monotonic",
                        lambda: clock["t"]), \
             mock.patch("time.sleep",
                        lambda s: sleeps.append(s)):
            for _ in range(5):
                limiter.acquire()
        self.assertEqual(sleeps, [])


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", headers=None,
                 json_error=False):
        self.status_code = status_code
        self._json = json_data
        self.text = text
        self.headers = headers or {}
        self._json_error = json_error

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        if self._json_error:
            # Mimic requests' JSONDecodeError (a ValueError subclass) on an
            # empty/malformed body.
            raise ValueError("No JSON could be decoded")
        return self._json


class FakeSession:
    """Returns queued FakeResponses (or raises queued exceptions) per call.

    If more than one outcome remains it pops the next; the last outcome
    repeats indefinitely (so a single 429 means 'always 429').
    """

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = []

    def _next(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def get(self, *args, **kwargs):
        return self._next(*args, **kwargs)

    def post(self, *args, **kwargs):
        return self._next(*args, **kwargs)


def _adapter_with(outcomes):
    adapter = EnsemblRESTAdapter()
    adapter._session = FakeSession(outcomes)
    return adapter


class TestRequestFailureMapping(TestCase):
    def setUp(self):
        # 429 path sleeps on Retry-After; patch it so tests stay fast.
        self._sleep_patch = mock.patch(
            "time.sleep", lambda s: None)
        self._sleep_patch.start()

    def tearDown(self):
        self._sleep_patch.stop()

    def test_429_exhausted_raises_ratelimit(self):
        adapter = _adapter_with([FakeResponse(429, headers={"Retry-After": "0"})])
        with self.assertRaises(EnsemblRateLimitError):
            adapter._request("/lookup/id/ENST1")

    def test_connection_error_raises_connection(self):
        adapter = _adapter_with([requests.exceptions.ConnectionError("boom")])
        with self.assertRaises(EnsemblConnectionError):
            adapter._request("/lookup/id/ENST1")

    def test_timeout_raises_connection(self):
        adapter = _adapter_with([requests.exceptions.Timeout("slow")])
        with self.assertRaises(EnsemblConnectionError):
            adapter._request("/lookup/id/ENST1")

    def test_retryerror_raises_base_rest_error(self):
        adapter = _adapter_with([requests.exceptions.RetryError("5xx exhausted")])
        with self.assertRaises(EnsemblRESTError) as ctx:
            adapter._request("/lookup/id/ENST1")
        self.assertNotIsInstance(
            ctx.exception, (EnsemblRateLimitError, EnsemblConnectionError))

    def test_non_retryable_error_status_raises_base_rest_error(self):
        # A non-2xx, non-429, non-400/404 status (e.g. 403) is a definitive
        # failure: raise the base error, not a rate-limit/connection subtype.
        adapter = _adapter_with([FakeResponse(403)])
        with self.assertRaises(EnsemblRESTError) as ctx:
            adapter._request("/lookup/id/FORBIDDEN")
        self.assertNotIsInstance(
            ctx.exception, (EnsemblRateLimitError, EnsemblConnectionError))

    def test_200_unparseable_json_returns_none(self):
        adapter = _adapter_with([FakeResponse(200, json_error=True)])
        self.assertIsNone(adapter._request("/lookup/id/EMPTY"))

    def test_400_returns_none(self):
        adapter = _adapter_with([FakeResponse(400)])
        self.assertIsNone(adapter._request("/lookup/id/BAD"))

    def test_404_returns_none(self):
        adapter = _adapter_with([FakeResponse(404)])
        self.assertIsNone(adapter._request("/lookup/id/MISSING"))

    def test_200_json_returns_parsed(self):
        adapter = _adapter_with([FakeResponse(200, json_data={"id": "ENST1"})])
        self.assertEqual(adapter._request("/lookup/id/ENST1"), {"id": "ENST1"})

    def test_200_text_returns_text(self):
        adapter = _adapter_with([FakeResponse(200, text="MEEPQS")])
        self.assertEqual(
            adapter._request("/sequence/id/ENSP1", content_type="text/plain"), "MEEPQS")

    def test_429_then_success(self):
        adapter = _adapter_with([
            FakeResponse(429, headers={"Retry-After": "0"}),
            FakeResponse(200, json_data={"id": "ENST1"}),
        ])
        self.assertEqual(adapter._request("/lookup/id/ENST1"), {"id": "ENST1"})
