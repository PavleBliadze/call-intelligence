"""
logger_setup.py
==============

Responsibility: configure logging once for the whole pipeline.

Logs go to the console and to run.log. Deliberately, full transcripts are
NEVER logged here - the run log must not become a store of private call
content. Only filenames, languages, and verdicts are logged.
"""

from __future__ import annotations

import contextlib
import logging

import config

_configured = False


def get_logger() -> logging.Logger:
    """Return the shared 'pipeline' logger, configuring it on first call."""
    global _configured
    logger = logging.getLogger("pipeline")
    if not _configured:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s")

        console = logging.StreamHandler()
        # Non-Latin text (e.g. a reason or a file name) must render on the
        # console.
        # The default Windows console codepage (cp1252) can't encode it, so
        # force the stream to UTF-8 where the platform allows it.
        with contextlib.suppress(AttributeError, ValueError):
            console.stream.reconfigure(encoding="utf-8", errors="replace")
        console.setFormatter(fmt)
        logger.addHandler(console)

        file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        _configured = True
    return logger
