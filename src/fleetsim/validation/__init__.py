"""fleetsim.validation — reproduce published cluster-trace results.

The v0.6 validation suite proves fleetsim's occupancy / queue-wait / JCT
numbers against reality: it downloads published traces (Helios, Philly,
Alibaba PAI), replays them, and asserts the papers' reported ratios and
distributions (validation plan §2).

Public surface
--------------
- :func:`fetch_trace` and the trace registry (``TRACE_REGISTRY``) —
  stdlib download + integrity gate + Git-LFS-pointer detection.
- Metric adapters (:func:`jct_over_all_terminal`, :func:`n_queuing_jobs`,
  :func:`gpu_time_by_status`) — recompute the papers' summary definitions
  from ``jobs.parquet`` where they diverge from ``summary.json``.
- ``convert_helios`` (in ``helios.py``) and ``per_vc_replay`` (in
  ``harness.py``) — the trace converter and per-VC replay harness,
  exposed here via LAZY import so importing this package never requires
  those later-phase modules to exist yet.

Importing this package has no side effects beyond importing the always-
present submodules (registry, fetch, adapters).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .adapters import (
    gpu_time_by_status,
    jct_over_all_terminal,
    n_queuing_jobs,
)
from .fetch import (
    IntegrityError,
    LFSPointerError,
    UngatedTraceError,
    fetch_trace,
)
from .registry import TRACE_REGISTRY, TraceSpec, get_spec

if TYPE_CHECKING:  # pragma: no cover - typing only; lazy at runtime
    from .harness import per_vc_replay, replay_canonical
    from .helios import convert_helios, month_window
    from .philly_status import status_split_by_count, status_split_by_gpu_time

__all__ = [
    # fetch + registry
    "fetch_trace",
    "TRACE_REGISTRY",
    "TraceSpec",
    "get_spec",
    "IntegrityError",
    "LFSPointerError",
    "UngatedTraceError",
    # adapters
    "jct_over_all_terminal",
    "n_queuing_jobs",
    "gpu_time_by_status",
    # later-phase (lazy)
    "convert_helios",
    "month_window",
    "status_split_by_count",
    "status_split_by_gpu_time",
    "per_vc_replay",
    "replay_canonical",
]

#: Names resolved lazily on first access so this package imports cleanly
#: before the later-phase modules land.  Maps public name -> (submodule,
#: attribute).
_LAZY: dict[str, tuple[str, str]] = {
    "convert_helios": (".helios", "convert_helios"),
    "month_window": (".helios", "month_window"),
    "status_split_by_count": (".philly_status", "status_split_by_count"),
    "status_split_by_gpu_time": (".philly_status", "status_split_by_gpu_time"),
    "per_vc_replay": (".harness", "per_vc_replay"),
    "replay_canonical": (".harness", "replay_canonical"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module, attr = target
    return getattr(importlib.import_module(module, __name__), attr)


def __dir__() -> list[str]:
    return sorted(__all__)
