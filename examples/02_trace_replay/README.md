# Example 02 — trace replay

Replays `sample_trace.csv` — a small **synthetic, hand-written** sample in
fleetsim's canonical trace schema (it is NOT real cluster data; see the
header comments in the file) — on an 8-node × 8-chip fleet under strict
FIFO.

```bash
fleetsim run scenario.yaml -o out
```

The canonical schema (DESIGN §10) is the union of the public traces:

    job_id, user, tenant, class, submit_time, num_chips, chip_type,
    num_nodes, duration_s, walltime_limit_s, final_status

`submit_time` is int microseconds since the trace epoch; `*_s` columns are
float seconds; `final_status` uses Helios's enum (COMPLETED, FAILED,
CANCELED, TIMEOUT, NODE_FAIL) and is replayed **verbatim** — compare the
`status` column of `out/jobs.parquet` against the CSV.

To replay a real trace (e.g. Microsoft Philly), convert it first:
`fleetsim.workload.philly.convert_philly()` maps the published JSON to
canonical rows, and `fleetsim.workload.trace.write_trace()` writes the
CSV; then point `workload.source` at it and scale `fleet` accordingly.

Trace `num_chips` need not obey the allocation grammar: any count above
the node size is rounded UP to the next whole-node multiple at load time
(DESIGN §4.1) — Philly's widest-attempt GPU sums (e.g. 8+4=12) would
otherwise be permanently unplaceable and starve a strict-FIFO queue.
Sub-node counts replay verbatim.
