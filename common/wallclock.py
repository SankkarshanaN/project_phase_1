import time


class WallclockBudget:
    """Tracks a hard wallclock cutoff for a training/collection stage.

    Collection scripts poll `.expired()` in their main loop and must save
    what they have and exit as soon as it returns True.
    """

    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        self.start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def remaining(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed())

    def expired(self) -> bool:
        return self.elapsed() >= self.max_seconds
