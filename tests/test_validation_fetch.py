"""Unit tests for the trace fetch layer + registry (v0.6, checklist
item 4).

NO NETWORK: these exercise the integrity gate, Git-LFS-pointer
detection, and cache-path resolution against tiny local fixtures — never
:func:`fetch_trace`'s download path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fleetsim.validation import fetch as F
from fleetsim.validation.registry import TRACE_REGISTRY, get_spec


# --------------------------------------------------------------------------
# Integrity gate
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_file(tmp_path: Path) -> Path:
    p = tmp_path / "artifact.bin"
    p.write_bytes(b"fleetsim validation fixture\n")
    return p


def test_sha256_gate_matches(tiny_file: Path):
    digest = hashlib.sha256(tiny_file.read_bytes()).hexdigest()
    assert F.verify_integrity(tiny_file, sha256=digest, name="t") == tiny_file


def test_sha256_gate_rejects_mismatch(tiny_file: Path):
    with pytest.raises(F.IntegrityError, match="SHA-256 mismatch"):
        F.verify_integrity(tiny_file, sha256="0" * 64, name="t")


def test_sha256_file_matches_hashlib(tiny_file: Path):
    assert F.sha256_file(tiny_file) == hashlib.sha256(tiny_file.read_bytes()).hexdigest()


def test_size_gate_matches_and_rejects(tiny_file: Path):
    size = tiny_file.stat().st_size
    assert F.verify_integrity(tiny_file, size=size, name="t") == tiny_file
    with pytest.raises(F.IntegrityError, match="size mismatch"):
        F.verify_integrity(tiny_file, size=size + 1, name="t")


def test_partial_sha256_is_documentation_and_falls_back_to_size(tiny_file: Path):
    """A non-full SHA-256 (Philly's documented ``prefix...suffix``) is
    ignored for gating; the size gate decides."""
    size = tiny_file.stat().st_size
    # Passes on correct size despite the bogus-looking partial hash...
    assert F.verify_integrity(
        tiny_file, sha256="2037ccf6...13d2c", size=size, name="philly"
    ) == tiny_file
    # ...and the partial hash never causes a hash mismatch (wrong size does).
    with pytest.raises(F.IntegrityError, match="size mismatch"):
        F.verify_integrity(
            tiny_file, sha256="2037ccf6...13d2c", size=size + 1, name="philly"
        )


def test_ungated_entry_refused(tiny_file: Path):
    with pytest.raises(F.UngatedTraceError):
        F.verify_integrity(tiny_file, name="pai")


# --------------------------------------------------------------------------
# Git-LFS pointer detection
# --------------------------------------------------------------------------


LFS_POINTER = (
    b"version https://git-lfs.github.com/spec/v1\n"
    b"oid sha256:4d7a...\nsize 1055988361\n"
)


def test_is_lfs_pointer_bytes_and_path(tmp_path: Path):
    ptr = tmp_path / "trace-data.tar.gz"
    ptr.write_bytes(LFS_POINTER)
    assert F.is_lfs_pointer(ptr) is True
    assert F.is_lfs_pointer(LFS_POINTER) is True
    # A real (non-pointer) payload is not flagged.
    real = tmp_path / "real.bin"
    real.write_bytes(b"\x1f\x8b\x08\x00 gzip magic and bytes")
    assert F.is_lfs_pointer(real) is False


def test_lfs_pointer_raises_with_pull_instruction(tmp_path: Path):
    ptr = tmp_path / "trace-data.tar.gz"
    ptr.write_bytes(LFS_POINTER)
    with pytest.raises(F.LFSPointerError) as exc:
        F.verify_integrity(ptr, size=len(LFS_POINTER), name="philly", lfs=True)
    msg = str(exc.value)
    assert "git lfs pull" in msg
    assert "pointer" in msg.lower()


def test_lfs_pointer_checked_before_size(tmp_path: Path):
    """Even when the pointer happens to satisfy the size gate, the LFS
    error wins (a pointer must never be accepted as the payload)."""
    ptr = tmp_path / "p"
    ptr.write_bytes(LFS_POINTER)
    with pytest.raises(F.LFSPointerError):
        F.verify_integrity(ptr, size=ptr.stat().st_size, name="philly")


# --------------------------------------------------------------------------
# Cache-path resolution
# --------------------------------------------------------------------------


def test_cache_root_precedence(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FLEETSIM_TRACE_CACHE", raising=False)
    # Default: ~/.cache/fleetsim/traces
    assert F.resolve_cache_root() == Path.home() / ".cache" / "fleetsim" / "traces"
    # Env override.
    monkeypatch.setenv("FLEETSIM_TRACE_CACHE", str(tmp_path / "env"))
    assert F.resolve_cache_root() == tmp_path / "env"
    # Explicit argument beats the env var.
    assert F.resolve_cache_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_cache_path_layout(tmp_path: Path):
    cp = F.cache_path("helios", cache_dir=tmp_path)
    assert cp == tmp_path / "helios" / "data.zip"
    cp2 = F.cache_path("philly", cache_dir=tmp_path)
    assert cp2 == tmp_path / "philly" / "trace-data.tar.gz"


def test_fetch_trace_returns_verified_cache_without_download(tmp_path: Path):
    """A pre-placed, valid cache file is returned as-is (no network)."""
    dest = F.cache_path("helios", cache_dir=tmp_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = b"x" * 100
    dest.write_bytes(payload)
    # Point the spec's size at our fixture so the gate passes.
    import dataclasses
    from unittest import mock

    patched = dataclasses.replace(get_spec("helios"), size=len(payload))
    with mock.patch.dict(TRACE_REGISTRY, {"helios": patched}):
        got = F.fetch_trace("helios", cache_dir=tmp_path)
    assert got == dest


def test_fetch_trace_refuses_mismatched_cache(tmp_path: Path):
    dest = F.cache_path("helios", cache_dir=tmp_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"x" * 7)  # wrong size vs registry's 36,437,672
    with pytest.raises(F.IntegrityError):
        F.fetch_trace("helios", cache_dir=tmp_path)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_registry_has_expected_traces():
    assert {"helios", "philly", "pai_task_table", "pai_machine_spec",
            "pai_instance_table"} <= set(TRACE_REGISTRY)


def test_registry_entries_carry_attribution():
    for name, spec in TRACE_REGISTRY.items():
        assert spec.name == name
        assert spec.url.startswith("http")
        assert spec.license and spec.citation and spec.attribution_url
        assert spec.extract_hint
        # Every entry has *some* integrity signal (full hash, size, or an
        # explicit human size_hint for the ungated opt-in traces).
        assert spec.sha256 or spec.size or spec.size_hint


def test_helios_is_size_gated_not_lfs():
    h = get_spec("helios")
    assert h.lfs is False
    assert h.size == 36_437_672
    assert h.uncompressed_size == 343_069_364
    assert "raw.githubusercontent.com" in h.url


def test_philly_is_lfs_and_size_gated():
    p = get_spec("philly")
    assert p.lfs is True
    assert p.size == 1_055_988_361
    assert p.sha256 and p.sha256.startswith("2037ccf6")


def test_get_spec_unknown_raises_valueerror():
    with pytest.raises(ValueError, match="unknown trace"):
        get_spec("does-not-exist")
