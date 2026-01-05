# Make phase1 a proper package and provide convenient re-exports.
# Also add a tiny bootstrap to ensure `python -m phase1.train` works reliably.

# Re-exports for convenience
from .config import Phase1Config  # type: ignore
from .model import Phase1CoconutModel  # type: ignore

# Backward-compatible alias expected by some callers
Phase1Model = Phase1CoconutModel

__all__ = [
    "Phase1Config",
    "Phase1CoconutModel",
    "Phase1Model",
]

# Bootstrap for module execution: when running as `python -m phase1.train`,
# ensure package-relative imports behave consistently.
if __name__ == "__main__":
    # Allow running this package directly for debug: `python -m phase1`
    # Delegates to the train entry if present.
    try:
        from .train import main as _main  # type: ignore
    except Exception:
        # Fallback: import and run train as a script-like entry
        import runpy
        runpy.run_module("phase1.train", run_name="__main__")
    else:
        _main()
