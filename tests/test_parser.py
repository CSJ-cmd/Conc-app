import pytest

from structural_safety_core_v6 import parse_readings_text, parse_ocr_readings_text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("54 56 55", [54.0, 56.0, 55.0]),
        ("54,56,55", [54.0, 56.0, 55.0]),
        ("54, 56, 55", [54.0, 56.0, 55.0]),
        ("54.5 56.0 55.5", [54.5, 56.0, 55.5]),
        ("54\n56\n55", [54.0, 56.0, 55.0]),
        ("[54, 56, 55]", [54.0, 56.0, 55.0]),
        (None, []),
        ("", []),
    ],
)
def test_manual_reading_parser(raw, expected):
    assert parse_readings_text(raw) == expected


def test_ocr_parser_keeps_ocr_specific_corrections():
    assert parse_ocr_readings_text("O 1 l | 58,4") == [0.0, 1.0, 1.0, 1.0, 58.4]
