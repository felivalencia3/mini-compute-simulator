"""Trace registry: where each published cluster trace lives, how big it
is, how to verify it, and the license/citation it must be attributed
under (v0.6 validation suite, DESIGN §5 of the validation plan).

Every entry is a :class:`TraceSpec`.  Integrity is gated by **either** a
full SHA-256 **or** a byte ``size`` (some upstreams publish one, some the
other; a couple publish neither and are opt-in / ungated).  A
``sha256`` that is not a full 64-hex digest (e.g. a documented
``"prefix...suffix"`` marker for a hash we carry for provenance but did
not recompute) is treated as documentation, and :mod:`fleetsim.validation.fetch`
falls back to the ``size`` gate — see :func:`fleetsim.validation.fetch.verify_integrity`.

Nothing here downloads; this module is pure data.  ``fetch.py`` consumes
it.  Keys are the trace names passed to
:func:`fleetsim.validation.fetch.fetch_trace`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TraceSpec", "TRACE_REGISTRY", "get_spec"]


@dataclass(frozen=True, slots=True)
class TraceSpec:
    """Everything :mod:`fleetsim.validation.fetch` needs to download and
    verify one published trace artifact, plus the attribution metadata
    the license requires.

    Fields
    ------
    name:
        Registry key (also the cache subdirectory name).
    url:
        Direct download URL for the single artifact this entry names.
    sha256:
        Full 64-hex SHA-256 of the downloaded artifact, or a documented
        non-full marker (provenance only — see the module docstring), or
        ``None`` when the artifact is size-gated instead.
    size:
        Exact byte size of the downloaded artifact (the integrity gate
        when ``sha256`` is absent/partial), or ``None`` if unknown.
    uncompressed_size:
        Byte size after extraction, when documented (informational).
    license:
        SPDX-ish license string the trace is released under.
    citation:
        Human citation string the attribution requires.
    attribution_url:
        Canonical source/repository URL for attribution.
    extract_hint:
        One-line human note on how to extract the artifact (what files
        it yields), surfaced in errors and the ``cite`` CLI.
    lfs:
        True when the upstream artifact is stored via Git-LFS, so a plain
        HTTP fetch of the raw path yields a ~135-byte pointer rather than
        the real bytes; ``fetch.py`` uses this to enrich the LFS-pointer
        error with the correct ``git lfs pull`` remediation.
    size_hint:
        Human-readable approximate size for entries with no exact
        integrity gate (opt-in traces whose exact bytes we do not carry).
    """

    name: str
    url: str
    license: str
    citation: str
    attribution_url: str
    extract_hint: str
    sha256: str | None = None
    size: int | None = None
    uncompressed_size: int | None = None
    lfs: bool = False
    size_hint: str | None = None


#: The known published traces.  See the v0.6 validation plan §5.
TRACE_REGISTRY: dict[str, TraceSpec] = {
    # Helios (SC '21) — a plain Git object (NOT LFS); size-gated because
    # the compressed archive's exact byte count is published and stable.
    "helios": TraceSpec(
        name="helios",
        url="https://raw.githubusercontent.com/S-Lab-System-Group/HeliosData/master/data.zip",
        size=36_437_672,
        uncompressed_size=343_069_364,
        license="CC-BY-4.0",
        citation=(
            "Hu et al., 'Characterization and Prediction of Deep Learning "
            "Workloads in Large-Scale GPU Datacenters', SC '21, "
            "DOI 10.1145/3458817.3476223"
        ),
        attribution_url="https://github.com/S-Lab-System-Group/HeliosData",
        extract_hint=(
            "unzip data.zip -> data/{Venus,Earth,Saturn,Uranus}/"
            "{cluster_log.csv,cluster_gpu_number.csv}"
        ),
        lfs=False,
    ),
    # Philly (USENIX ATC '19) — Git-LFS: a plain clone / raw fetch yields
    # a ~135-byte pointer, so `git lfs pull` is required.  We carry the
    # documented SHA-256 prefix..suffix for provenance but gate on the
    # exact byte size (the full digest is not recomputed here); fetch.py
    # detects the LFS pointer and raises with the pull instruction.
    "philly": TraceSpec(
        name="philly",
        url="https://raw.githubusercontent.com/msr-fiddle/philly-traces/master/trace-data.tar.gz",
        sha256="2037ccf6...13d2c",  # documented prefix..suffix (provenance; size-gated)
        size=1_055_988_361,
        license="CC-BY-4.0",
        citation=(
            "Jeon et al., 'Analysis of Large-Scale Multi-Tenant GPU "
            "Clusters for DNN Training Workloads', USENIX ATC '19"
        ),
        attribution_url="https://github.com/msr-fiddle/philly-traces",
        extract_hint=(
            "git lfs install && git lfs pull; tar xzf trace-data.tar.gz -> "
            "trace-data/cluster_job_log (JSON array, no file extension)"
        ),
        lfs=True,
    ),
    # Alibaba PAI (NSDI '22) — free public use.  Three headerless CSV
    # tarballs; exact byte sizes are published in the README but not
    # recomputed here, so these are opt-in and carry only size_hints
    # (no exact integrity gate).  Column names come from the README when
    # convert_pai (v0.6 stretch / v0.7) parses them.
    "pai_task_table": TraceSpec(
        name="pai_task_table",
        url="https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_task_table.tar.gz",
        license="free public use (Alibaba Cluster Trace Program)",
        citation="Weng et al., 'MLaaS in the Wild', NSDI '22 (weng2022mlaas)",
        attribution_url=(
            "https://github.com/alibaba/clusterdata/tree/master/"
            "cluster-trace-gpu-v2020"
        ),
        extract_hint="tar xzf -> pai_task_table.csv (headerless; columns from README)",
        lfs=False,
        size_hint="~34 MB",
    ),
    "pai_machine_spec": TraceSpec(
        name="pai_machine_spec",
        url="https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_spec.tar.gz",
        license="free public use (Alibaba Cluster Trace Program)",
        citation="Weng et al., 'MLaaS in the Wild', NSDI '22 (weng2022mlaas)",
        attribution_url=(
            "https://github.com/alibaba/clusterdata/tree/master/"
            "cluster-trace-gpu-v2020"
        ),
        extract_hint="tar xzf -> pai_machine_spec.csv (headerless; columns from README)",
        lfs=False,
        size_hint="~32 KB",
    ),
    "pai_instance_table": TraceSpec(
        name="pai_instance_table",
        url="https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_instance_table.tar.gz",
        license="free public use (Alibaba Cluster Trace Program)",
        citation="Weng et al., 'MLaaS in the Wild', NSDI '22 (weng2022mlaas)",
        attribution_url=(
            "https://github.com/alibaba/clusterdata/tree/master/"
            "cluster-trace-gpu-v2020"
        ),
        extract_hint="tar xzf -> pai_instance_table.csv (headerless; columns from README)",
        lfs=False,
        size_hint="~663 MB",
    ),
}


def get_spec(name: str) -> TraceSpec:
    """The :class:`TraceSpec` registered under ``name``.

    Raises ``ValueError`` (listing the known names) for an unknown trace,
    so a typo surfaces as a clean config error rather than a ``KeyError``.
    """
    spec = TRACE_REGISTRY.get(name)
    if spec is None:
        known = ", ".join(sorted(TRACE_REGISTRY)) or "none"
        raise ValueError(f"unknown trace {name!r} (registered: {known})")
    return spec
