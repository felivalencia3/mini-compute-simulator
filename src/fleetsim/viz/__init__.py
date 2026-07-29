"""fleetsim.viz — the v0.3 visualizer data pipeline.

Turns a ``fleetsim run`` output directory (``summary.json`` +
``jobs.parquet`` + ``timeseries.parquet`` + optional ``stints.parquet``)
into one JSON-serializable "replay model" dict that the self-contained
HTML report renders.  See :mod:`fleetsim.viz.data` for the pinned model
schema and :mod:`fleetsim.viz.render` for the single-file HTML app.
"""

from .data import build_viz_model, to_json
from .render import render_html

__all__ = ["build_viz_model", "render_html", "to_json"]
