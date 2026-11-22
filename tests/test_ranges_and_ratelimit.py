from itertools import pairwise

import pytest

from mono_agent.mono.ranges import MAX_WINDOW_S, split_window, subtract
from mono_agent.mono.ratelimit import TokenBucket


def test_split_window_covers_range_without_gaps_or_overlap():
    start, end = 1_000, 1_000 + 90 * 86_400
    windows = split_window(start, end)
    assert windows[0][0] == start and windows[-1][1] == end
    assert all(b - a <= MAX_WINDOW_S for a, b in windows)
    assert all(nxt[0] == prev[1] + 1 for prev, nxt in pairwise(windows))


def test_split_window_short_range_is_one_window():
    assert split_window(10, 20) == [(10, 20)]


def test_split_window_rejects_reversed_range():
    with pytest.raises(ValueError):
        split_window(20, 10)


@pytest.mark.parametrize(
    ("covered", "gaps"),
    [
        ([], [(0, 100)]),
        ([(0, 100)], []),
        ([(20, 40)], [(0, 19), (41, 100)]),
        ([(-50, 10), (90, 200)], [(11, 89)]),
        ([(30, 60), (10, 40)], [(0, 9), (61, 100)]),
    ],
)
def test_subtract(covered, gaps):
    assert subtract((0, 100), covered) == gaps


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_token_bucket_one_call_per_period():
    clock = FakeClock()
    bucket = TokenBucket(capacity=1, period=60, clock=clock, sleep=clock.sleep)
    assert bucket.acquire() == 0  # first call is free
    assert bucket.acquire() == pytest.approx(60)  # second waits a full period
    clock.now += 30
    assert bucket.acquire() == pytest.approx(30)  # partial refill shortens the wait
    assert clock.now == pytest.approx(120)


def test_token_bucket_does_not_accumulate_beyond_capacity():
    clock = FakeClock()
    bucket = TokenBucket(capacity=1, period=60, clock=clock, sleep=clock.sleep)
    clock.now += 3600  # idle for an hour
    assert bucket.acquire() == 0
    assert bucket.acquire() == pytest.approx(60)
