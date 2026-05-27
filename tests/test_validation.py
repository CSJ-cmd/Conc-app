import numpy as np
import pytest

from structural_safety_core_v6 import calculate_strength, validate_rebound_inputs


@pytest.mark.parametrize(
    "label, result",
    [
        ("NaN 측정값 차단", calculate_strength([50] * 19 + [np.nan], angle=0, days=3000, design_fck=24, core_coeff=1.0)),
        ("inf 측정값 차단", calculate_strength([50] * 19 + [np.inf], angle=0, days=3000, design_fck=24, core_coeff=1.0)),
        ("허용 범위 밖 측정값 차단", calculate_strength([50] * 19 + [5], angle=0, days=3000, design_fck=24, core_coeff=1.0)),
        ("잘못된 각도 차단", calculate_strength([50] * 20, angle=30, days=3000, design_fck=24, core_coeff=1.0)),
        ("재령 0 차단", calculate_strength([50] * 20, angle=0, days=0, design_fck=24, core_coeff=1.0)),
        ("설계강도 0 차단", calculate_strength([50] * 20, angle=0, days=3000, design_fck=0, core_coeff=1.0)),
        ("Ct 0 차단", calculate_strength([50] * 20, angle=0, days=3000, design_fck=24, core_coeff=0)),
        ("알 수 없는 공식 차단", calculate_strength([50] * 20, angle=0, days=3000, design_fck=24, selected_formulas=["없는공식"], core_coeff=1.0)),
    ],
)
def test_invalid_rebound_inputs_are_rejected(label, result):
    ok, detail = result
    assert not ok, label
    assert isinstance(detail, str)
    assert detail


def test_boolean_values_are_not_accepted_as_numbers():
    ok, detail = validate_rebound_inputs([50] * 19 + [True], angle=0, days=3000, design_fck=24, core_coeff=1)
    assert not ok
    assert "True/False" in detail
