# -*- coding: utf-8 -*-
import logging
import os
import time

from .utils.logging import setup_logger

# Fallback before we can safely read canonical constant definitions.
LOG_LEVEL_ENV = "AWOMOPAW_LOG_LEVEL"
if LOG_LEVEL_ENV not in os.environ and "QWENPAW_LOG_LEVEL" in os.environ:
    os.environ[LOG_LEVEL_ENV] = os.environ["QWENPAW_LOG_LEVEL"]

_bootstrap_err: Exception | None = None
try:
    # Load persisted env vars before importing modules that read env-backed
    # constants at import time (e.g., WORKING_DIR).
    from .envs import load_envs_into_environ

    load_envs_into_environ()
except Exception as exc:
    # Best effort: package import should not fail if env bootstrap fails.
    _bootstrap_err = exc

_t0 = time.perf_counter()
setup_logger(
    os.environ.get(
        LOG_LEVEL_ENV,
        os.environ.get("QWENPAW_LOG_LEVEL", "info"),
    ),
)
if _bootstrap_err is not None:
    logging.getLogger(__name__).warning(
        "awomopaw: failed to load persisted envs on init: %s",
        _bootstrap_err,
    )
logging.getLogger(__name__).debug(
    "%.3fs package init",
    time.perf_counter() - _t0,
)
