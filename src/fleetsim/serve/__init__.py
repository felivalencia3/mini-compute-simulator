"""fleetsim.serve — the ``fleetsim serve`` local web app (v0.5).

Pure-stdlib HTTP server (``http.server``) around the existing run/viz
pipeline: browse past runs in a workspace, open any run as the
interactive 2D report, submit new scenario YAML and watch live progress,
and (later phases) a three.js 3D fleet replay.  Local-first: binds
loopback by default, server-generated run ids, path containment on every
client-supplied name, ``yaml.safe_load`` only, JSON errors only.

See :mod:`fleetsim.serve.server` for the pinned route contract and
:mod:`fleetsim.serve.runs` for workspace/run-lifecycle semantics.
"""

from .runs import RunManager
from .server import FleetsimHTTPServer, serve

__all__ = ["FleetsimHTTPServer", "RunManager", "serve"]
