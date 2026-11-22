"""Token bucket: `capacity` calls burst, refilled at one token per `period` seconds."""

import threading
import time
from collections.abc import Callable


class TokenBucket:
    def __init__(
        self,
        capacity: int = 1,
        period: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.capacity = capacity
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._tokens = float(capacity)
        self._last = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) / self.period)
        self._last = now

    def acquire(self) -> float:
        """Block until a token is available; return how long we waited."""
        with self._lock:  # held while sleeping on purpose: callers queue in order
            self._refill()
            waited = 0.0
            if self._tokens < 1:
                waited = (1 - self._tokens) * self.period
                self._sleep(waited)
                self._refill()
            self._tokens -= 1
            return waited


class Unlimited(TokenBucket):
    """No-op bucket for the synthetic fake API, where there is nothing to protect."""

    def acquire(self) -> float:
        return 0.0
