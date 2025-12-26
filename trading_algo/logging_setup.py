from __future__ import annotations

import logging
import os
import sys


def configure_logging(
    *,
    level: int = logging.INFO,
    log_file: str | None = None,
    console: bool = True,
) -> None:
    handlers: list[logging.Handler] = []
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    if console:
        handlers.append(logging.StreamHandler(stream=sys.stderr))

    if log_file:
        log_file = str(log_file)
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    if not handlers:
        # Never leave logging unconfigured.
        handlers.append(logging.NullHandler())

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
