"""Tests for fleetsim.units: constants, parse_duration, format_duration."""

import pytest

from fleetsim.units import DAY, HOUR, MIN, MS, S, US, WEEK, format_duration, parse_duration


class TestConstants:
    def test_ladder(self):
        assert US == 1
        assert MS == 1_000
        assert S == 1_000_000
        assert MIN == 60 * S
        assert HOUR == 60 * MIN
        assert DAY == 24 * HOUR
        assert WEEK == 7 * DAY

    def test_all_int(self):
        for const in (US, MS, S, MIN, HOUR, DAY, WEEK):
            assert type(const) is int


class TestParseDuration:
    def test_int_is_seconds(self):
        assert parse_duration(60) == 60 * S
        assert parse_duration(0) == 0

    def test_float_is_seconds(self):
        assert parse_duration(1.5) == 1_500_000
        assert parse_duration(0.000001) == 1  # one microsecond

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("60s", 60 * S),
            ("2m", 2 * MIN),
            ("1.5h", HOUR + 30 * MIN),
            ("14d", 14 * DAY),
            ("1w", WEEK),
            ("500ms", 500 * MS),
            ("250us", 250 * US),
            ("0.5d", 12 * HOUR),
            ("0s", 0),
            ("90", 90 * S),  # bare number string = seconds
            ("  2m  ", 2 * MIN),  # whitespace tolerated
            ("1e3s", 1000 * S),  # scientific notation
        ],
    )
    def test_suffixes(self, text, expected):
        assert parse_duration(text) == expected

    def test_returns_int(self):
        assert type(parse_duration("1.5h")) is int
        assert type(parse_duration(2.5)) is int

    @pytest.mark.parametrize("bad", ["", "abc", "5x", "s", "1h30m", "--5s", "nan"])
    def test_malformed_raises_value_error(self, bad):
        with pytest.raises(ValueError):
            parse_duration(bad)

    @pytest.mark.parametrize("bad", [-1, -0.5, "-5s", "-1"])
    def test_negative_raises(self, bad):
        with pytest.raises(ValueError):
            parse_duration(bad)

    def test_non_finite_raises(self):
        with pytest.raises(ValueError):
            parse_duration(float("inf"))

    def test_bool_rejected(self):
        # YAML `true` must not silently become 1 second
        with pytest.raises(TypeError):
            parse_duration(True)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            parse_duration([60])


class TestFormatDuration:
    @pytest.mark.parametrize(
        "us,expected",
        [
            (0, "0s"),
            (60 * S, "1m"),
            (90 * S, "90s"),  # not a whole number of minutes
            (HOUR, "1h"),
            (90 * MIN, "90m"),  # 1.5h formats via the largest exact unit
            (3 * DAY, "3d"),
            (14 * DAY, "2w"),  # same value, larger unit
            (1500 * MS, "1500ms"),
            (123, "123us"),
            (2 * MIN, "2m"),
        ],
    )
    def test_exact_unit(self, us, expected):
        assert format_duration(us) == expected

    @pytest.mark.parametrize(
        "value",
        [0, 1, 999, 1234, MS, S, MIN, HOUR, DAY, WEEK, 90 * MIN, 5 * S + 1, 42 * DAY],
    )
    def test_round_trip(self, value):
        assert parse_duration(format_duration(value)) == value

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            format_duration(-1)

    def test_non_int_raises(self):
        with pytest.raises(TypeError):
            format_duration(1.5)
        with pytest.raises(TypeError):
            format_duration(True)
