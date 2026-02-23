from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class StageUpdate:
    stage: int
    advanced: bool
    forced_stage1_exit: bool
    improved: bool
    patience_counter: int
    best_val_acc: Optional[float]


class StageManager:
    def __init__(
        self,
        *,
        min_delta: float,
        stage_patience: Sequence[int],
        max_steps_first_stage: int,
    ) -> None:
        if len(stage_patience) != 8:
            raise ValueError("stage_patience must provide 8 values for stages 1..8.")
        self.min_delta = float(min_delta)
        self.stage_patience = tuple(int(x) for x in stage_patience)
        self.max_steps_first_stage = int(max_steps_first_stage)

        self.current_stage = 1
        self.best_val_acc: Optional[float] = None
        self.no_improve_count = 0

    def _advance_stage(self) -> bool:
        if self.current_stage >= 8:
            return False
        self.current_stage += 1
        self.best_val_acc = None
        self.no_improve_count = 0
        return True

    def force_stage1_exit_if_needed(self, optimizer_steps: int) -> bool:
        if self.current_stage != 1:
            return False
        if int(optimizer_steps) < self.max_steps_first_stage:
            return False
        return self._advance_stage()

    def update(self, *, val_acc: float, optimizer_steps: int) -> StageUpdate:
        del optimizer_steps
        forced = False
        improved = False
        advanced = False

        if self.best_val_acc is None:
            self.best_val_acc = float(val_acc)
            self.no_improve_count = 0
            return StageUpdate(
                stage=self.current_stage,
                advanced=False,
                forced_stage1_exit=False,
                improved=True,
                patience_counter=0,
                best_val_acc=self.best_val_acc,
            )

        if float(val_acc) >= float(self.best_val_acc) + self.min_delta:
            self.best_val_acc = float(val_acc)
            self.no_improve_count = 0
            improved = True
        else:
            self.no_improve_count += 1
            patience = self.stage_patience[self.current_stage - 1]
            if self.no_improve_count >= patience:
                advanced = self._advance_stage()

        return StageUpdate(
            stage=self.current_stage,
            advanced=advanced,
            forced_stage1_exit=forced,
            improved=improved,
            patience_counter=self.no_improve_count,
            best_val_acc=self.best_val_acc,
        )
