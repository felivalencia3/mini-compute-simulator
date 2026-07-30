"""fleetsim — a discrete-event simulator for ML accelerator fleets.

Public API (stable within v0.1):

- :func:`run_scenario` — one-call pipeline: scenario in, summary dict out.
- :func:`load_scenario` / :func:`validate` / :class:`ScenarioError` —
  config loading and validation.
- :class:`Scheduler` / :class:`PlacementPolicy` / :func:`register` /
  :func:`get_scheduler` — the pluggable-policy surface (out-of-tree
  schedulers may also register via the ``fleetsim.schedulers``
  entry-point group).
- :func:`get_placement` / :func:`placement_names` — the *where*-axis
  surface (v0.7).  Policies are also selectable by name from YAML with
  ``scheduler: {params: {placement: best_fit}}``; the default stays
  ``first_fit``.  The policy classes themselves live in
  ``fleetsim.schedulers.placement``.
- :class:`Job` / :class:`JobView` — the job model and its
  scheduler-visible snapshot.
- Actions: :class:`Place`, :class:`Preempt` (with :class:`PreemptMode`),
  and the :data:`Action` union.

Everything else is importable from its submodule (``fleetsim.config``,
``fleetsim.fleet``, ``fleetsim.engine``, ``fleetsim.workload``,
``fleetsim.metrics``) but is not re-exported here.
"""

from .api import run_scenario
from .config import ScenarioError, load_scenario, validate
from .model import Job, PreemptMode
from .schedulers.base import (
    Action,
    JobView,
    Place,
    PlacementPolicy,
    Preempt,
    Scheduler,
    get_scheduler,
    register,
)
from .schedulers.placement import get_placement, placement_names

__version__ = "0.8.0"

__all__ = [
    "__version__",
    "run_scenario",
    "load_scenario",
    "validate",
    "ScenarioError",
    "Scheduler",
    "PlacementPolicy",
    "register",
    "get_scheduler",
    "get_placement",
    "placement_names",
    "Job",
    "JobView",
    "Place",
    "Preempt",
    "PreemptMode",
    "Action",
]
