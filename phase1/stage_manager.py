from typing import Optional, Tuple

class StageManager:
    def __init__(
        self,
        stage_patience,
        *,
        min_delta: float = 0.01,
    ):
        if len(stage_patience) != 8:
            raise ValueError("Expected 8 patience values for stages 1..8")

        self.stage_patience = stage_patience
        self.min_delta = min_delta

        self.current_stage = 1
        self.best_val_acc: Optional[float] = None
        self.no_improve_count = 0

        self._last_eval_id: Optional[int] = None
        self.done = False
        self.skip_to_stage = {4:8}

    def reset(self):
        self.current_stage = 1
        self.best_val_acc = None
        self.no_improve_count = 0
        self._last_eval_id = None
        self.done = False

    def update_on_evaluation(
        self,
        eval_id: int,
        val_acc: float,
    ) -> Tuple[bool, bool]:
        """
        Returns (advanced, done)
        """
        if self.done:
            return False, True

        # Enforce one update per evaluation window
        if self._last_eval_id == eval_id:
            return False, self.done
        self._last_eval_id = eval_id

        # First evaluation initializes baseline
        if self.best_val_acc is None:
            self.best_val_acc = val_acc
            self.no_improve_count = 0
            return False, False

        # Check improvement
        if val_acc >= self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.no_improve_count = 0
            print(
                f"[Stage {self.current_stage}] Improvement detected: "
                f"{val_acc:.4f} (best updated)"
            )
            return False, False

        # No improvement
        self.no_improve_count += 1
        patience = self.stage_patience[self.current_stage - 1]

        print(
            f"[Stage {self.current_stage}] No improvement "
            f"({self.no_improve_count}/{patience}) "
            f"val_acc={val_acc:.4f}, best={self.best_val_acc:.4f}"
        )

        if self.no_improve_count >= patience:
            if self.current_stage < 8:
                next_stage = self.skip_to_stage.get(self.current_stage, self.current_stage + 1)
                self.current_stage = min(8, int(next_stage))

                self.no_improve_count = 0
                self.best_val_acc = None  # reset baseline for new stage
                print(f"➡️ Advancing to stage {self.current_stage}")
                return True, False
            else:
                self.done = True
                print("🏁 Stage 8 stagnated → training complete")
                return True, True

        return False, False

    def move_to_next_stage(self) -> None:
        self.current_stage += 1
        self.no_improve_count = 0
        self.best_val_acc = None

