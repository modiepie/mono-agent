"""Pure helpers for time ranges in unix seconds. Ranges are inclusive [start, end]."""

# The statement endpoint accepts at most 31 days + 1 hour per request.
MAX_WINDOW_S = 31 * 24 * 3600 + 3600  # 2_682_000

Range = tuple[int, int]


def split_window(start: int, end: int, max_window: int = MAX_WINDOW_S) -> list[Range]:
    """Split [start, end] into consecutive windows no longer than max_window."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    windows = []
    while start <= end:
        w_end = min(start + max_window, end)
        windows.append((start, w_end))
        start = w_end + 1
    return windows


def subtract(target: Range, covered: list[Range]) -> list[Range]:
    """Parts of target not covered by any range in covered."""
    gaps = []
    cursor, end = target
    for c_start, c_end in sorted(covered):
        if c_end < cursor or c_start > end:
            continue
        if c_start > cursor:
            gaps.append((cursor, c_start - 1))
        cursor = max(cursor, c_end + 1)
        if cursor > end:
            return gaps
    if cursor <= end:
        gaps.append((cursor, end))
    return gaps
