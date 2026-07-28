# Example 03 — an out-of-tree custom scheduler

A complete, minimal plugin package labs can copy as a starting point. It
implements **smallest-job-first** (order pending jobs by ascending chip
count, place first-fit, best-effort) and exposes it to fleetsim through
the `fleetsim.schedulers` entry-point group — no fleetsim code changes.

```bash
pip install -e .                       # from this directory
fleetsim run scenario.yaml -o out      # scheduler: {name: smallest_first}
```

How it works:

1. `fleetsim_smallest_first.py` subclasses `fleetsim.Scheduler`, sorts
   `view.pending()` by `(chips, submit_time, id)`, and emits
   `Place(job.id, placement)` for every job `view.find_placement()` can
   fit — the view's tentative-reservation semantics guarantee the loop
   never double-books capacity.
2. `pyproject.toml` declares the entry point
   `smallest_first = "fleetsim_smallest_first:SmallestFirstScheduler"`
   under `[project.entry-points."fleetsim.schedulers"]`; when a scenario
   names `smallest_first`, fleetsim's registry loads it on demand.
3. The `@register("smallest_first")` decorator also runs on import, so
   `import fleetsim_smallest_first` works in scripts and notebooks too.

Scheduler params in the scenario are passed to `__init__` as keyword
arguments — try `scheduler: {name: smallest_first, params: {strict: true}}`.
