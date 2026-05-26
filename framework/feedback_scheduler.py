from typing import Optional, Tuple


class FeedbackScheduler:
    """Adaptive gate for expensive feedback/refine in single-shot PRO.

    Gating uses **per-request repeat index** (``repeat_idx`` from the driver
    when ``repeat_shots_per_request`` is enabled). Each ``attack_request`` call
    is one self-contained shot; repeat index counts shots across a request, not
    nested dialogue rounds inside a shot.
    """

    def __init__(
        self,
        every_n_repeats: int = 2,
        min_quality: float = 0.35,
        min_delta: float = 0.02,
        cooldown_repeats: int = 1,
        request_time_budget_ms: float = 5000.0,
    ):
        self.every_n_repeats = max(1, int(every_n_repeats))
        self.min_quality = float(min_quality)
        self.min_delta = float(min_delta)
        self.cooldown_repeats = max(0, int(cooldown_repeats))
        self.request_time_budget_ms = max(0.0, float(request_time_budget_ms))
        self._last_feedback_repeat_idx = -10**9

    def _delta_ok(self, curr: float, prev: Optional[float]) -> bool:
        if prev is None:
            return True
        return (float(curr) - float(prev)) >= self.min_delta

    def _periodic_ok(self, repeat_idx: int) -> bool:
        return int(repeat_idx) % self.every_n_repeats == 0

    def _cooldown_ok(self, repeat_idx: int) -> bool:
        return (int(repeat_idx) - int(self._last_feedback_repeat_idx)) > self.cooldown_repeats

    def should_run(
        self,
        *,
        repeat_idx: int,
        best_failed_score: float,
        prev_best_failed_score: Optional[float],
        request_feedback_spent_ms: float,
    ) -> Tuple[bool, str]:
        if float(request_feedback_spent_ms) >= self.request_time_budget_ms:
            return False, "budget_exceeded"
        if not self._cooldown_ok(repeat_idx):
            return False, "cooldown_active"

        quality_ok = float(best_failed_score) >= self.min_quality
        periodic_ok = self._periodic_ok(repeat_idx)
        delta_ok = self._delta_ok(best_failed_score, prev_best_failed_score)

        if (quality_ok and delta_ok) or periodic_ok:
            self._last_feedback_repeat_idx = int(repeat_idx)
            if periodic_ok and not quality_ok:
                return True, "periodic_gate"
            if quality_ok and delta_ok:
                return True, "quality_and_delta_gate"
            return True, "quality_gate"
        return False, "gating_not_satisfied"
