import pytest

from structural_safety_core_v6 import (
    REBOUND_POINT_POLICY_EXACT_20,
    REBOUND_POINT_POLICY_MIN_20,
    calculate_strength,
    get_discard_limit_for_policy,
    get_recommended_formulas,
    normalize_rebound_point_policy,
)


def test_formula_auto_recommendation_by_design_strength():
    assert get_recommended_formulas(24) == ["일본건축", "일본재료"]
    assert get_recommended_formulas(39.9) == ["일본건축", "일본재료"]
    assert get_recommended_formulas(40) == ["과기부", "권영웅", "KALIS"]


def test_formula_mode_auto_and_manual_are_separated():
    ok_a, res_a = calculate_strength([50] * 20, angle=0, days=3000, design_fck=24, selected_formulas=None, core_coeff=1.0)
    ok_b, res_b = calculate_strength([50] * 20, angle=0, days=3000, design_fck=40, selected_formulas=None, core_coeff=1.0)
    ok_c, res_c = calculate_strength([50] * 20, angle=0, days=3000, design_fck=24, selected_formulas=["KALIS"], core_coeff=1.0)
    ok_d, res_d = calculate_strength([50] * 20, angle=0, days=3000, design_fck=24, selected_formulas=[], core_coeff=1.0)

    assert ok_a, res_a
    assert res_a["Formula_Mode"] == "자동추천"
    assert res_a["Applied_Formulas"] == ["일본건축", "일본재료"]

    assert ok_b, res_b
    assert res_b["Formula_Mode"] == "자동추천"
    assert res_b["Applied_Formulas"] == ["과기부", "권영웅", "KALIS"]

    assert ok_c, res_c
    assert res_c["Formula_Mode"] == "수동선택"
    assert res_c["Applied_Formulas"] == ["KALIS"]

    assert not ok_d
    assert "1개 이상" in str(res_d)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("정확히 20개", REBOUND_POINT_POLICY_EXACT_20),
        ("정확히20", REBOUND_POINT_POLICY_EXACT_20),
        ("exact_20", REBOUND_POINT_POLICY_EXACT_20),
        ("20개 이상 허용", REBOUND_POINT_POLICY_MIN_20),
        ("20개이상", REBOUND_POINT_POLICY_MIN_20),
        ("min_20", REBOUND_POINT_POLICY_MIN_20),
    ],
)
def test_point_policy_aliases(raw, expected):
    assert normalize_rebound_point_policy(raw) == expected


def test_exact_20_policy_requires_exactly_20_readings():
    ok_20, res_20 = calculate_strength([50] * 20, angle=0, days=3000, design_fck=24, core_coeff=1.0, point_count_policy=REBOUND_POINT_POLICY_EXACT_20)
    ok_19, res_19 = calculate_strength([50] * 19, angle=0, days=3000, design_fck=24, core_coeff=1.0, point_count_policy=REBOUND_POINT_POLICY_EXACT_20)
    ok_21, res_21 = calculate_strength([50] * 21, angle=0, days=3000, design_fck=24, core_coeff=1.0, point_count_policy=REBOUND_POINT_POLICY_EXACT_20)

    assert ok_20, res_20
    assert res_20["Point_Count_Policy"] == REBOUND_POINT_POLICY_EXACT_20
    assert not ok_19
    assert "정확히 20개" in str(res_19)
    assert not ok_21
    assert "정확히 20개" in str(res_21)


def test_min_20_policy_allows_more_than_20_and_uses_ratio_discard_limit():
    ok_21, res_21 = calculate_strength([50] * 21, angle=0, days=3000, design_fck=24, core_coeff=1.0, point_count_policy=REBOUND_POINT_POLICY_MIN_20)
    ok_5_discard, res_5_discard = calculate_strength([50] * 19 + [20] * 5, angle=0, days=3000, design_fck=24, core_coeff=1.0, point_count_policy=REBOUND_POINT_POLICY_MIN_20)
    ok_6_discard, res_6_discard = calculate_strength([50] * 18 + [20] * 6, angle=0, days=3000, design_fck=24, core_coeff=1.0, point_count_policy=REBOUND_POINT_POLICY_MIN_20)

    assert get_discard_limit_for_policy(24, REBOUND_POINT_POLICY_MIN_20) == 6

    assert ok_21, res_21
    assert res_21["Point_Count_Policy"] == REBOUND_POINT_POLICY_MIN_20
    assert res_21["N"] == 21

    assert ok_5_discard, res_5_discard
    assert res_5_discard["Discard"] == 5
    assert res_5_discard["Discard_Limit"] == 6

    assert not ok_6_discard
    assert "기각" in str(res_6_discard)
