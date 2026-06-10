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
        with mock.patch("epytope.IO.EnsemblRESTAdapter.time.monotonic",
                        lambda: clock["t"]), \
             mock.patch("epytope.IO.EnsemblRESTAdapter.time.sleep", fake_sleep):
            limiter.acquire()   # 1st call: fits
            limiter.acquire()   # 2nd call: fits
            limiter.acquire()   # 3rd call: window full -> must wait 10s

        self.assertEqual(sleeps, [10.0])

    def test_no_block_under_limit(self):
        clock = {"t": 500.0}
        sleeps = []
        limiter = _RateLimiter([(5, 1.0)])
        with mock.patch("epytope.IO.EnsemblRESTAdapter.time.monotonic",
                        lambda: clock["t"]), \
             mock.patch("epytope.IO.EnsemblRESTAdapter.time.sleep",
                        lambda s: sleeps.append(s)):
            for _ in range(5):
                limiter.acquire()
        self.assertEqual(sleeps, [])
