"""Keep YASA spindle detection importable in the headless frozen backend."""

from __future__ import annotations

import sys
from types import ModuleType


def _unsupported_plot(*_args, **_kwargs):
    raise RuntimeError("YASA plotting is unavailable in the headless PyHFO backend.")


plotting = ModuleType("yasa.plotting")
setattr(plotting, "__all__", ["plot_hypnogram"])
setattr(plotting, "plot_hypnogram", _unsupported_plot)
sys.modules.setdefault("yasa.plotting", plotting)

staging = ModuleType("yasa.staging")
setattr(staging, "__all__", [])
sys.modules.setdefault("yasa.staging", staging)
