# Proxy module: re-export Python's stdlib logging to avoid shadowing issues with third-party libs.
# This keeps the package path `logging` available for our submodules (e.g., logging.episode_logger)
# while ensuring that `import logging` refers to the real stdlib logging API.
from LMMS_logging import *  # noqa: F401,F403

