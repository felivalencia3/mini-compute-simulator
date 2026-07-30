"""Download + integrity-gate published traces into a local cache.

STDLIB ONLY.  Uses ``urllib.request`` + ``hashlib`` — no new runtime
dependency (the whole point of keeping the validation fetch layer free of
``requests``/``boto3``).  Nothing here is imported on the hot simulation
path; it exists purely for the opt-in validation replays.

Cache layout::

    <cache_root>/<trace_name>/<artifact_filename>

where ``<cache_root>`` is, in order of precedence: the explicit
``cache_dir`` argument, ``$FLEETSIM_TRACE_CACHE``, else
``~/.cache/fleetsim/traces``.

Integrity is gated per :class:`~fleetsim.validation.registry.TraceSpec`:
a **full** 64-hex SHA-256 is verified when present; otherwise the exact
byte ``size`` is verified.  A ``sha256`` that is not a full digest (a
documented ``"prefix...suffix"`` provenance marker) is ignored for
gating and the ``size`` gate is used instead.  A trace with neither a
full hash nor a size is *ungated* and :func:`fetch_trace` refuses to
download it (opt-in traces the caller must place manually).

Git-LFS safety: a raw HTTP fetch of an LFS-backed artifact yields a tiny
pointer file beginning ``version https://git-lfs...`` rather than the
real bytes.  :func:`fetch_trace` detects that pointer and raises
:class:`LFSPointerError` with the ``git lfs pull`` remediation instead of
caching the pointer.

DETERMINISM / SAFETY: a cached artifact that fails its gate is refused
(never silently re-downloaded or used).  No wall-clock, no randomness.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.request
from pathlib import Path

from .registry import TraceSpec, get_spec

__all__ = [
    "fetch_trace",
    "verify_integrity",
    "is_lfs_pointer",
    "sha256_file",
    "resolve_cache_root",
    "cache_path",
    "IntegrityError",
    "LFSPointerError",
    "UngatedTraceError",
    "LFS_POINTER_PREFIX",
]

#: The first bytes of a Git-LFS pointer file (LFS v1 spec line).
LFS_POINTER_PREFIX = b"version https://git-lfs"

_FULL_SHA256 = re.compile(r"\A[0-9a-fA-F]{64}\Z")

_READ_CHUNK = 1 << 20  # 1 MiB streaming chunk for hashing / copying


class IntegrityError(Exception):
    """A cached or downloaded artifact failed its size/SHA-256 gate."""


class LFSPointerError(Exception):
    """The artifact on disk is a Git-LFS pointer, not the real payload."""


class UngatedTraceError(Exception):
    """The registry entry has neither a full SHA-256 nor a size, so the
    artifact cannot be integrity-verified and is refused for auto-fetch."""


def _is_full_sha256(value: str | None) -> bool:
    """True iff ``value`` is a full 64-hex SHA-256 digest (documented
    ``prefix...suffix`` markers and ``None`` are not)."""
    return bool(value) and bool(_FULL_SHA256.match(value or ""))


def sha256_file(path: str | Path) -> str:
    """Stream ``path`` through SHA-256 and return the lowercase hex
    digest (constant memory — never loads the whole file)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(_READ_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def is_lfs_pointer(source: str | Path | bytes) -> bool:
    """True iff ``source`` (a path or a bytes head) begins with the
    Git-LFS pointer signature.  A path reads only the first 64 bytes."""
    if isinstance(source, (bytes, bytearray)):
        head = bytes(source[:64])
    else:
        with Path(source).open("rb") as f:
            head = f.read(64)
    return head.startswith(LFS_POINTER_PREFIX)


def verify_integrity(
    path: str | Path,
    *,
    sha256: str | None = None,
    size: int | None = None,
    name: str = "trace",
    lfs: bool = False,
) -> Path:
    """Verify the artifact at ``path`` against a size and/or SHA-256 gate.

    Order: LFS-pointer check first (a pointer can never satisfy a real
    gate and deserves the specific remediation), then a **full** SHA-256
    when one is supplied, else the exact ``size``.  A non-full ``sha256``
    (documented prefix marker) is ignored and the ``size`` gate is used;
    when neither a full hash nor a size is available, raises
    :class:`UngatedTraceError`.

    Returns ``path`` (as a :class:`Path`) on success; raises
    :class:`LFSPointerError`, :class:`IntegrityError`, or
    :class:`UngatedTraceError` otherwise.  A failure message reads like a
    lab-notebook entry: it names the trace, the expected value, and the
    measured one.
    """
    p = Path(path)
    if is_lfs_pointer(p):
        raise LFSPointerError(_lfs_message(name, p, lfs=lfs))

    has_hash = _is_full_sha256(sha256)
    actual_size = p.stat().st_size
    if size is not None and actual_size != size:
        raise IntegrityError(
            f"{name}: size mismatch for {p} — expected {size} B, "
            f"measured {actual_size} B (refusing to use a mismatched file; "
            f"delete it and re-fetch)"
        )
    if has_hash:
        actual = sha256_file(p)
        if actual.lower() != (sha256 or "").lower():
            raise IntegrityError(
                f"{name}: SHA-256 mismatch for {p} — expected {sha256}, "
                f"measured {actual} (refusing to use a mismatched file; "
                f"delete it and re-fetch)"
            )
    elif size is None:
        raise UngatedTraceError(
            f"{name}: registry entry has no full SHA-256 and no size, so "
            f"{p} cannot be integrity-verified; place the artifact manually "
            f"and gate it before use"
        )
    return p


def _lfs_message(name: str, path: Path, *, lfs: bool) -> str:
    hint = (
        " This artifact is Git-LFS backed: a plain clone or raw HTTP fetch "
        "yields the pointer, not the payload."
        if lfs
        else ""
    )
    return (
        f"{name}: {path} is a Git-LFS pointer, not the real trace data."
        f"{hint} Run `git lfs install && git lfs pull` in the source "
        f"repository (then copy the resolved file into the cache), or fetch "
        f"the artifact through Git-LFS — fleetsim will not cache the pointer."
    )


def resolve_cache_root(cache_dir: str | Path | None = None) -> Path:
    """The cache root: explicit ``cache_dir`` > ``$FLEETSIM_TRACE_CACHE``
    > ``~/.cache/fleetsim/traces`` (``~`` expanded).  No directory is
    created here."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    env = os.environ.get("FLEETSIM_TRACE_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "fleetsim" / "traces"


def _artifact_filename(spec: TraceSpec) -> str:
    """The artifact's on-disk filename (basename of the URL)."""
    name = spec.url.rstrip("/").rsplit("/", 1)[-1]
    return name or spec.name


def cache_path(name: str, cache_dir: str | Path | None = None) -> Path:
    """Where the artifact for trace ``name`` lives (or would live) in the
    cache.  Does not create anything or check existence."""
    spec = get_spec(name)
    return resolve_cache_root(cache_dir) / spec.name / _artifact_filename(spec)


def _verify_spec(path: Path, spec: TraceSpec) -> Path:
    return verify_integrity(
        path,
        sha256=spec.sha256,
        size=spec.size,
        name=spec.name,
        lfs=spec.lfs,
    )


def fetch_trace(name: str, cache_dir: str | Path | None = None) -> Path:
    """Return a verified local path to the artifact for trace ``name``,
    downloading it into the cache if it is not already present.

    A cached artifact is re-verified (a mismatched cache is refused, never
    silently reused).  A fresh download streams to a ``.part`` temp file,
    is verified (LFS-pointer + size/SHA-256), and only then atomically
    moved into place — so a failed or interrupted fetch never leaves a
    bad file at the final path.

    Raises ``ValueError`` for an unknown trace, :class:`LFSPointerError`
    when the fetched bytes are a Git-LFS pointer, :class:`IntegrityError`
    on a size/hash mismatch, :class:`UngatedTraceError` when the registry
    entry has no integrity gate, and ``OSError``/``urllib`` errors on I/O
    or network failure.
    """
    spec = get_spec(name)
    dest = cache_path(name, cache_dir)
    if dest.exists():
        return _verify_spec(dest, spec)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(spec.url) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(_READ_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
        _verify_spec(tmp, spec)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)
    return dest
