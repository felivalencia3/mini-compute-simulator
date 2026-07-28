"""Tests for fleetsim.engine.rng: named-stream independence, caching, and
per-stream override reseeding."""

import pytest

from fleetsim.engine.rng import RngStreams


def _draws(gen, n=8):
    return [gen.random() for _ in range(n)]


def test_same_seed_same_name_same_sequence():
    a = RngStreams(42)
    b = RngStreams(42)
    assert _draws(a.stream("arrivals")) == _draws(b.stream("arrivals"))


def test_different_names_are_independent_streams():
    r = RngStreams(42)
    assert _draws(r.stream("arrivals")) != _draws(r.stream("failures"))


def test_different_seeds_differ():
    assert _draws(RngStreams(0).stream("arrivals")) != _draws(
        RngStreams(1).stream("arrivals")
    )


def test_stream_is_cached_by_name():
    r = RngStreams(7)
    assert r.stream("failures") is r.stream("failures")


def test_draws_on_one_stream_do_not_perturb_another():
    # Determinism contract: enabling/consuming one subsystem's stream never
    # changes another stream's sequence.
    plain = RngStreams(123)
    noisy = RngStreams(123)
    _ = _draws(noisy.stream("failures"), 100)  # heavy interleaved use
    a = _draws(plain.stream("arrivals"))
    b = _draws(noisy.stream("arrivals"))
    assert a == b


def test_creation_order_is_irrelevant():
    r1 = RngStreams(9)
    r1.stream("a")
    seq_b1 = _draws(r1.stream("b"))
    r2 = RngStreams(9)
    seq_b2 = _draws(r2.stream("b"))  # "b" created first here
    assert seq_b1 == seq_b2


def test_overrides_reseed_only_named_stream():
    base = RngStreams(0)
    tweaked = RngStreams(0, overrides={"failures": 999})
    assert _draws(base.stream("arrivals")) == _draws(tweaked.stream("arrivals"))
    assert _draws(base.stream("failures")) != _draws(tweaked.stream("failures"))


def test_override_matches_plain_seed():
    # An override behaves exactly as if that stream's root seed were the
    # override value.
    assert _draws(RngStreams(5, overrides={"x": 77}).stream("x")) == _draws(
        RngStreams(77).stream("x")
    )


def test_seed_property_and_type_check():
    assert RngStreams(3).seed == 3
    with pytest.raises(TypeError):
        RngStreams(True)
    with pytest.raises(TypeError):
        RngStreams("42")
