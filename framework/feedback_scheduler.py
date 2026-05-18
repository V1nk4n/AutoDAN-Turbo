from typing import Tuple


class FeedbackScheduler:
    """Gate expensive feedback/refine between PRO repeats.

    Runs feedback when ``repeat_idx % every_n_repeats == 0``, with cooldown between calls.
  """

    def __init__(
        self,
        every_n_repeats: int = 2,
        cooldown_repeats: int = 1,
    ):
        self.every_n_repeats = max(1, int(every_n_repeats))
        self.cooldown_repeats = max(0, int(cooldown_repeats))
        self._last_feedback_repeat_idx = -10**9

    def reset_for_request(self) -> None:
        """Clear cooldown state so each harmful request gets its own feedback schedule."""
        self._last_feedback_repeat_idx = -10**9

    def _periodic_ok(self, repeat_idx: int) -> bool:
        return int(repeat_idx) % self.every_n_repeats == 0

    def _cooldown_ok(self, repeat_idx: int) -> bool:
        return (int(repeat_idx) - int(self._last_feedback_repeat_idx)) > self.cooldown_repeats

    def should_run(self, *, repeat_idx: int) -> Tuple[bool, str]:
        if not self._cooldown_ok(repeat_idx):
            return False, "cooldown_active"
        if not self._periodic_ok(repeat_idx):
            return False, "not_periodic_repeat"
        self._last_feedback_repeat_idx = int(repeat_idx)
        return True, "periodic_gate"
