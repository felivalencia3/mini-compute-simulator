"""fleetsim.serve — the ``fleetsim serve`` local web app (v0.5, v0.8).

Pure-stdlib HTTP server (``http.server``) around the existing run/viz
pipeline: browse past runs in a workspace, open any run as the
interactive 2D report, submit new scenario YAML and watch live progress,
and a three.js 3D fleet replay.  Local-first: binds loopback by default,
server-generated run ids, path containment on every client-supplied
name, ``yaml.safe_load`` only, JSON errors only.

v0.8 adds parallel worker PROCESSES (:func:`default_max_workers`,
``serve --workers``), parameter sweeps (:class:`SweepManager`), and a live
replay stream that lets a client build the fleet map while a run is still
executing.

See :mod:`fleetsim.serve.server` for the pinned route contract,
:mod:`fleetsim.serve.runs` for workspace/run-lifecycle semantics and the
live-stream cursor contract, and :mod:`fleetsim.serve.sweeps` for sweep
expansion rules.

Calling :func:`serve` from your own script requires an
``if __name__ == "__main__":`` guard — worker processes re-execute the
main module (see :class:`UnguardedMainError`).
"""

from .runs import RunManager, UnguardedMainError, default_max_workers
from .server import FleetsimHTTPServer, serve
from .sweeps import SweepManager

__all__ = [
    "FleetsimHTTPServer",
    "RunManager",
    "SweepManager",
    "UnguardedMainError",
    "default_max_workers",
    "serve",
]
