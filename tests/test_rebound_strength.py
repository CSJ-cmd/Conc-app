import pytest

from structural_safety_core_v6 import (
    calculate_strength,
    get_age_coefficient,
    get_angle_correction,
)


READINGS_TC1 = [
    58.4, 57.0, 61.8, 61.2, 60.6,
    58.9, 59.9, 58.9, 58.2, 57.8,
    61.5, 60.1, 64.1, 57.9, 59.3,
    56.8, 57.1, 58.0, 58.4, 58.0,
]


def test_excel_reference_case_matches_expected_values():
    ok, res = calculate_strength(
        READINGS_TC1,
        angle=90,
        days=3000,
        design_fck=40.0,
        selected_formulas=["일본재료", "일본건축", "과기부"],
        core_coeff=1.0,
        require_20_points=True,
    )

    assert ok, res
    assert res["R_avg"] == pytest.approx(59.195, abs=1e-3)
    assert res["Angle_Corr"] == pytest.approx(-3.680913945, abs=1e-6)
    assert res["R0"] == pytest.approx(55.514086055, abs=1e-6)
    assert res["Age_Coeff"] == pytest.approx(0.63, abs=1e-12)
    assert res["Formulas"]["일본재료"] == pytest.approx(33.0768202086, abs=1e-6)
    assert res["Formulas"]["일본건축"] == pytest.approx(31.194309588372, abs=1e-6)
    assert res["Formulas"]["과기부"] == pytest.approx(45.132810978528, abs=1e-6)


def test_two_outliers_are_discarded_but_test_remains_valid():
    ok, res = calculate_strength([50] * 18 + [10, 90], angle=0, days=3000, design_fck=24, core_coeff=1.0)
    assert ok, res
    assert res["Discard"] == 2


def test_five_outliers_in_exact_20_policy_invalidates_test():
    ok, res = calculate_strength([50] * 15 + [10, 90, 10, 90, 10], angle=0, days=3000, design_fck=24, core_coeff=1.0)
    assert not ok
    assert "시험 무효" in str(res)


def test_core_coefficient_scales_formula_results():
    ok_a, res_a = calculate_strength(READINGS_TC1, angle=90, days=3000, design_fck=40, core_coeff=1.0)
    ok_b, res_b = calculate_strength(READINGS_TC1, angle=90, days=3000, design_fck=40, core_coeff=1.10)

    assert ok_a, res_a
    assert ok_b, res_b
    assert res_b["Formulas"]["과기부"] == pytest.approx(res_a["Formulas"]["과기부"] * 1.10, abs=1e-6)


@pytest.mark.parametrize(
    "days, expected",
    [
        (10, 1.55),
        (28, 1.00),
        (3000, 0.63),
        (4000, 0.63),
        (5, 1.55),
        (75, 0.825),
    ],
)
def test_age_coefficient_interpolation(days, expected):
    assert get_age_coefficient(days) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "angle, expected",
    [
        (90, (-0.0018 * 50 * 50) + (0.2455 * 50) - 11.906),
        (45, (-0.0026 * 50 * 50) + (0.2563 * 50) - 9.24),
        (0, 0.0),
        (-45, (-0.0007 * 50 * 50) + (0.0129 * 50) + 3.14),
        (-90, (-0.0009 * 50 * 50) + (0.0094 * 50) + 4.48),
    ],
)
def test_angle_correction_formulas(angle, expected):
    assert get_angle_correction(50, angle) == pytest.approx(expected, abs=1e-12)
