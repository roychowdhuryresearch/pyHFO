"""Locations for persistent, user-owned PyHFO data."""

import os
from pathlib import Path
from uuid import uuid4


def pyhfo_home() -> Path:
    return Path(os.environ.get("PYHFO_HOME") or Path.home() / ".pyhfo").expanduser().resolve()


def export_root() -> Path:
    return Path(os.environ.get("PYHFO_AGENT_EXPORT_DIR") or pyhfo_home() / "exports").expanduser().resolve()


def new_run_directory(kind: str) -> Path:
    """Give each implicit output its own directory, including repeated or concurrent runs."""
    return export_root() / f"{kind}-{uuid4().hex}"
