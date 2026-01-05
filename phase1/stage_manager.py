"""
Manages stage progression and gating based on validation accuracy.
- Enforces evaluation cadence via eval_id so a stage can advance at most once per evaluation cycle.
- Adds hysteresis: requires N consecutive passes over the threshold to advance.
- Defines terminal semantics for stage 8: once threshold met with stability, marks done and never regresses.

Threshold semantics:
- thresholds is a list of 8 floats. Index i maps to Stage (i+1):
  thresholds[0] -> Stage 1, ..., thresholds[7] -> Stage 8.
- thresholds must be non-decreasing (monotonic):
  thresholds[i] <= thresholds[i+1] for all i in [0..6].
  This prevents later stages being easier than earlier stages and avoids stalling.

NOTE:
- update_on_evaluation() is the ONLY supported way to advance stages.
- advance() exists for backward compatibility and testing; do not use it in real training loops.
"""
from typing import List, Optional, Tuple
import warnings

class StageManager:
    def __init__(self, thresholds: List[float], *, consecutive_passes: int = 1):
        if len(thresholds) != 8:
            raise ValueError("Expected 8 stage thresholds for stages 1..8")
        # Enforce monotonic non-decreasing thresholds to ensure curriculum consistency.
        for i in range(len(thresholds) - 1):
            if thresholds[i] > thresholds[i + 1]:
                raise ValueError(
                    f"Thresholds must be non-decreasing; found thresholds[{i}]={thresholds[i]} > thresholds[{i+1}]={thresholds[i+1]}"
                )
        if consecutive_passes < 1:
            raise ValueError("consecutive_passes must be >= 1")
        self.thresholds = thresholds
        self.current_stage = 1
        # Hysteresis: require N consecutive evaluations meeting threshold before advancing.
        self.consecutive_required = consecutive_passes
        self._consec_pass_count = 0
        # Cadence control: only advance once per evaluation window.
        self._last_eval_id: Optional[int] = None
        # Terminal semantics: once stage 8 achieved with stability, training is done.
        self.done = False
        # Safety: manual advance is disabled unless armed by a qualifying evaluation.
        self._advance_armed = False

    def _meets_threshold(self, val_acc: float) -> bool:
        idx = self.current_stage - 1
        meets_threshold = val_acc >= self.thresholds[idx]
        print(f'Validation accuracy {val_acc:.4f} {"meets" if meets_threshold else "does not meet"} threshold {self.thresholds[idx]:.4f} for stage {self.current_stage}.')
        return meets_threshold

    def can_advance(self, val_acc: float) -> bool:
        """
        DEPRECATED: Use update_on_evaluation().
        This method does not enforce evaluation cadence and should not be used
        for stage transitions. It applies only the hysteresis check and may
        diverge from the supported path.
        """
        warnings.warn(
            "StageManager.can_advance() is deprecated. Use update_on_evaluation(eval_id, val_acc) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.done:
            return False
        # Determine if this evaluation hits threshold; do not mutate counters.
        if self._meets_threshold(val_acc):
            # It would advance if this pass completes the sequence.
            return (self._consec_pass_count + 1) >= self.consecutive_required
        else:
            return False

    def advance(self) -> None:
        """
        Legacy manual advance. Safe-guarded: only advances if armed by a prior
        evaluation that met stability. Prefer update_on_evaluation() for all
        training flows.
        """
        warnings.warn(
            "StageManager.advance() is legacy and discouraged. Prefer update_on_evaluation(eval_id, val_acc).",
            UserWarning,
            stacklevel=2,
        )
        if self.done:
            return
        if not self._advance_armed:
            # Not armed by evaluation; ignore manual advance to ensure safety.
            return
        self._advance_armed = False
        if self.current_stage < 8:
            self.current_stage += 1
            # Reset hysteresis counter for the new stage.
            self._consec_pass_count = 0
        else:
            # Stage 8 reached; mark done explicitly.
            self.done = True

    def reset(self) -> None:
        self.current_stage = 1
        self._consec_pass_count = 0
        self._last_eval_id = None
        self.done = False
        self._advance_armed = False

    def update_on_evaluation(self, eval_id: int, val_acc: float) -> Tuple[bool, bool]:
        """
        Primary API: call once per evaluation cycle.
        - Enforces cadence: repeated calls with the same eval_id do not cause multiple advances.
        - Applies hysteresis: requires consecutive passes to advance.
        - Handles terminal semantics: when stage 8 is achieved, sets done=True and never regresses.

        Returns (advanced, done).
        advanced: True if the stage advanced (or terminalized at stage 8) during this call.
        done: True if training is complete (stage 8 satisfied).
        """
        if self.done:
            return False, True

        # Prevent multiple advances in the same evaluation window.
        if self._last_eval_id == eval_id:
            return False, self.done
        self._last_eval_id = eval_id

        if self._meets_threshold(val_acc):
            self._consec_pass_count += 1
        else:
            # Reset stability on failure to meet threshold.
            self._consec_pass_count = 0

        advanced = False
        if self._consec_pass_count >= self.consecutive_required:
            if self.current_stage < 8:
                self.current_stage += 1
                advanced = True
                # Reset for the new stage.
                self._consec_pass_count = 0
                # Arm manual advance only after next qualifying eval; currently auto-advanced.
                self._advance_armed = False
            else:
                # Achieved stability at final stage -> terminal success.
                self.done = True
                advanced = True
                self._advance_armed = False
        else:
            # Not yet stable; disarm manual advance.
            self._advance_armed = False
        return advanced, self.done
