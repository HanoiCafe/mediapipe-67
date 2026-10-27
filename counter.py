from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SixSevenCounter:
    threshold: float = 0.06
    min_beat_interval: float = 0.08
    smoothing: float = 0.55
    lost_timeout: float = 0.4

    beats: int = 0
    _side: int = field(default=0, repr=False)
    _left_y: float | None = field(default=None, repr=False)
    _right_y: float | None = field(default=None, repr=False)
    _last_beat_t: float = field(default=float("-inf"), repr=False)
    _missing_since: float | None = field(default=None, repr=False)

    def reset(self) -> None:
        self.beats = 0
        self._forget_hands()
        self._last_beat_t = float("-inf")

    @property
    def pairs(self) -> int:
        return self.beats // 2

    @property
    def last_label(self) -> str:
        return "6" if self.beats % 2 == 1 else "7"

    def update(self, left_y: float | None, right_y: float | None, now: float) -> bool:
        if left_y is None or right_y is None:
            if self._missing_since is None:
                self._missing_since = now
            elif now - self._missing_since > self.lost_timeout:
                self._forget_hands()
            return False
        self._missing_since = None

        a = self.smoothing
        self._left_y = left_y if self._left_y is None else a * left_y + (1 - a) * self._left_y
        self._right_y = right_y if self._right_y is None else a * right_y + (1 - a) * self._right_y

        diff = self._right_y - self._left_y
        if diff > self.threshold:
            wanted = 1
        elif diff < -self.threshold:
            wanted = -1
        else:
            return False

        if self._side == 0:
            self._side = wanted
            return False
        if wanted == self._side:
            return False
        if now - self._last_beat_t < self.min_beat_interval:
            return False

        self._side = wanted
        self._last_beat_t = now
        self.beats += 1
        return True

    def _forget_hands(self) -> None:
        self._side = 0
        self._left_y = None
        self._right_y = None
        self._missing_since = None
