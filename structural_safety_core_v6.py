"""
구조물 안전진단 앱 v6 - 핵심 계산 로직 모듈

이 파일은 Streamlit UI와 분리된 순수 계산/검증 함수만 담습니다.
pytest 자동 테스트는 이 모듈을 대상으로 실행하므로, 앱 화면을 띄우지 않고도
반발경도 계산 로직의 회귀(regression)를 확인할 수 있습니다.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

import numpy as np


# =========================================================
# 반발경도 계산 입력 검증 기준
# =========================================================
ALLOWED_REBOUND_ANGLES = {-90, -45, 0, 45, 90}
REBOUND_READING_MIN = 10.0
REBOUND_READING_MAX = 100.0
REBOUND_FORMULA_OPTIONS = ["일본재료", "일본건축", "과기부", "권영웅", "KALIS"]
REBOUND_FORMULA_NAMES = set(REBOUND_FORMULA_OPTIONS)
REBOUND_FORMULA_RECOMMEND_THRESHOLD = 40.0

REBOUND_POINT_POLICY_EXACT_20 = "exact_20"
REBOUND_POINT_POLICY_MIN_20 = "min_20"
REBOUND_POINT_POLICY_NO_MINIMUM = "no_minimum"  # 내부/테스트용
DEFAULT_REBOUND_POINT_POLICY = REBOUND_POINT_POLICY_EXACT_20
REBOUND_DISCARD_COUNT_LIMIT_20 = 5
REBOUND_DISCARD_RATIO_LIMIT = 0.25

REBOUND_POINT_POLICY_OPTIONS = {
    REBOUND_POINT_POLICY_EXACT_20: {
        "label": "정확히 20개",
        "short_label": "정확히20",
        "description": "1개소당 측정값을 정확히 20개로 제한합니다. 기각값이 5개 이상이면 시험 무효입니다.",
    },
    REBOUND_POINT_POLICY_MIN_20: {
        "label": "20개 이상 허용",
        "short_label": "20개이상",
        "description": "측정값을 20개 이상 허용합니다. 기각 기준은 20개 중 5개에 해당하는 25% 이상으로 판정합니다.",
    },
    REBOUND_POINT_POLICY_NO_MINIMUM: {
        "label": "측정점수 제한 없음",
        "short_label": "제한없음",
        "description": "내부 검증용 정책입니다. 일반 현장 계산에서는 사용하지 않는 것을 권장합니다.",
    },
}

REBOUND_POINT_POLICY_LABEL_TO_KEY: dict[str, str] = {}
for _policy_key, _policy_meta in REBOUND_POINT_POLICY_OPTIONS.items():
    REBOUND_POINT_POLICY_LABEL_TO_KEY[_policy_key] = _policy_key
    REBOUND_POINT_POLICY_LABEL_TO_KEY[_policy_meta["label"]] = _policy_key
    REBOUND_POINT_POLICY_LABEL_TO_KEY[_policy_meta["short_label"]] = _policy_key

REBOUND_POINT_POLICY_LABEL_TO_KEY.update({
    "정확히 20": REBOUND_POINT_POLICY_EXACT_20,
    "정확히20개": REBOUND_POINT_POLICY_EXACT_20,
    "20": REBOUND_POINT_POLICY_EXACT_20,
    "exact": REBOUND_POINT_POLICY_EXACT_20,
    "exact20": REBOUND_POINT_POLICY_EXACT_20,
    "exact_20": REBOUND_POINT_POLICY_EXACT_20,
    "20개 이상": REBOUND_POINT_POLICY_MIN_20,
    "20개이상허용": REBOUND_POINT_POLICY_MIN_20,
    "최소20": REBOUND_POINT_POLICY_MIN_20,
    "min20": REBOUND_POINT_POLICY_MIN_20,
    "min_20": REBOUND_POINT_POLICY_MIN_20,
    "at_least_20": REBOUND_POINT_POLICY_MIN_20,
    "제한 없음": REBOUND_POINT_POLICY_NO_MINIMUM,
    "제한없음": REBOUND_POINT_POLICY_NO_MINIMUM,
    "none": REBOUND_POINT_POLICY_NO_MINIMUM,
    "no_minimum": REBOUND_POINT_POLICY_NO_MINIMUM,
})


def normalize_rebound_point_policy(point_count_policy: Any = None, require_20_points: bool = True) -> str:
    """
    측정점수 정책을 내부 key로 정규화합니다.

    기존 호출부 호환성:
    - point_count_policy가 None이고 require_20_points=True이면 기본값 "정확히 20개"를 사용합니다.
    - point_count_policy가 None이고 require_20_points=False이면 내부용 "제한 없음"으로 처리합니다.
    """
    if point_count_policy is None:
        return DEFAULT_REBOUND_POINT_POLICY if require_20_points else REBOUND_POINT_POLICY_NO_MINIMUM

    policy_text = str(point_count_policy).strip()
    if not policy_text:
        return DEFAULT_REBOUND_POINT_POLICY if require_20_points else REBOUND_POINT_POLICY_NO_MINIMUM

    policy_key = REBOUND_POINT_POLICY_LABEL_TO_KEY.get(policy_text)
    if policy_key:
        return policy_key

    normalized = policy_text.lower().replace(" ", "").replace("-", "_")
    policy_key = REBOUND_POINT_POLICY_LABEL_TO_KEY.get(normalized)
    if policy_key:
        return policy_key

    allowed = ", ".join(
        meta["label"]
        for key, meta in REBOUND_POINT_POLICY_OPTIONS.items()
        if key != REBOUND_POINT_POLICY_NO_MINIMUM
    )
    raise ValueError(f"알 수 없는 측정점수 정책입니다: {point_count_policy!r}. 허용값: {allowed}")


def get_rebound_point_policy_label(point_count_policy: Any) -> str:
    policy_key = normalize_rebound_point_policy(point_count_policy)
    return REBOUND_POINT_POLICY_OPTIONS[policy_key]["label"]


def get_rebound_point_policy_description(point_count_policy: Any) -> str:
    policy_key = normalize_rebound_point_policy(point_count_policy)
    return REBOUND_POINT_POLICY_OPTIONS[policy_key]["description"]


def get_rebound_point_policy_short_label(point_count_policy: Any) -> str:
    policy_key = normalize_rebound_point_policy(point_count_policy)
    return REBOUND_POINT_POLICY_OPTIONS[policy_key]["short_label"]


def get_discard_limit_for_policy(point_count: int, point_count_policy: Any) -> int:
    """측정점수 정책별 기각 무효 기준 개수를 반환합니다."""
    n = int(point_count)
    policy_key = normalize_rebound_point_policy(point_count_policy)

    if policy_key == REBOUND_POINT_POLICY_EXACT_20:
        return REBOUND_DISCARD_COUNT_LIMIT_20

    if policy_key in (REBOUND_POINT_POLICY_MIN_20, REBOUND_POINT_POLICY_NO_MINIMUM):
        return max(1, int(math.ceil(n * REBOUND_DISCARD_RATIO_LIMIT)))

    return REBOUND_DISCARD_COUNT_LIMIT_20


def get_recommended_formulas(design_fck: Any) -> list[str]:
    """
    설계강도 기준 평균 산정 공식 자동추천.

    - 40MPa 미만: 일본건축/일본재료
    - 40MPa 이상: 과기부/권영웅/KALIS
    """
    try:
        fck = float(design_fck)
    except (TypeError, ValueError):
        fck = 24.0

    if not np.isfinite(fck):
        fck = 24.0

    if fck < REBOUND_FORMULA_RECOMMEND_THRESHOLD:
        return ["일본건축", "일본재료"]
    return ["과기부", "권영웅", "KALIS"]


def get_recommended_formula_description(design_fck: Any) -> str:
    """UI/보고서 표시용 자동추천 설명 문구를 반환합니다."""
    formulas = get_recommended_formulas(design_fck)
    try:
        fck = float(design_fck)
    except (TypeError, ValueError):
        fck = 24.0

    range_label = (
        f"{REBOUND_FORMULA_RECOMMEND_THRESHOLD:g}MPa 미만 일반강도 기준"
        if fck < REBOUND_FORMULA_RECOMMEND_THRESHOLD
        else f"{REBOUND_FORMULA_RECOMMEND_THRESHOLD:g}MPa 이상 고강도 기준"
    )
    return f"{range_label}: {', '.join(formulas)}"


# =========================================================
# 입력 파서
# =========================================================
def _normalize_ocr_token(text: Any) -> str:
    """OCR 오인식 문자를 숫자 파싱 친화적으로 정규화합니다."""
    text = str(text)
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)

    replacements = {
        'O': '0', 'o': '0',
        'I': '1', 'l': '1', '|': '1',
        ',': '.', ';': '.',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.strip()


def _normalize_manual_reading_text(raw_text: Any) -> str:
    """
    사람이 직접 입력한 측정값 문자열을 파싱하기 쉽게 정리합니다.

    - 수동 입력에서 쉼표(,)는 기본적으로 숫자 구분자입니다.
    - OCR 오인식 보정은 수행하지 않습니다.
    - 소수점은 점(.) 사용을 권장합니다.
    """
    if raw_text is None:
        return ""

    text = str(raw_text)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    text = re.sub(r'(?<=\d),(?=\d)', ' ', text)
    text = re.sub(r'[;\t\r\n/]+', ' ', text)
    text = re.sub(r'[\[\]\(\)\{\}]', ' ', text)
    return text.strip()


def parse_readings_text(raw_text: Any) -> list[float]:
    """
    수동 입력/엑셀 입력/텍스트 입력에서 숫자 목록을 파싱합니다.
    """
    text = _normalize_manual_reading_text(raw_text)
    if not text:
        return []

    tokens = re.findall(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)', text)
    vals: list[float] = []
    for token in tokens:
        try:
            value = float(token)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            vals.append(value)
    return vals


def parse_ocr_readings_text(raw_text: Any) -> list[float]:
    """OCR 원문에서 숫자 목록을 파싱할 때 사용하는 함수입니다."""
    if raw_text is None:
        return []

    normalized = _normalize_ocr_token(str(raw_text))
    tokens = re.findall(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)', normalized)

    vals: list[float] = []
    for token in tokens:
        try:
            value = float(token)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            vals.append(value)
    return vals


def _coerce_finite_float(value: Any, field_name: str) -> tuple[bool, float | str]:
    """계산에 쓰는 입력값을 유한한 float로 변환합니다."""
    if isinstance(value, (bool, np.bool_)):
        return False, f"{field_name}은(는) True/False가 아니라 숫자로 입력해야 합니다."

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False, f"{field_name}은(는) 숫자여야 합니다. 입력값: {value!r}"

    if not np.isfinite(number):
        return False, f"{field_name}에 NaN 또는 무한대 값이 포함되어 있습니다."

    return True, number


def validate_rebound_inputs(
    readings: Iterable[Any] | None,
    angle: Any,
    days: Any,
    design_fck: Any = 24.0,
    selected_formulas: Iterable[str] | str | None = None,
    core_coeff: Any = 1.0,
) -> tuple[bool, dict[str, Any] | str]:
    """
    반발경도 계산 전 입력값을 검증하고 계산 가능한 형태로 정규화합니다.
    """
    if readings is None:
        return False, "데이터 없음: 측정값 목록이 비어 있습니다."

    if isinstance(readings, str):
        return False, "측정값은 숫자 리스트여야 합니다. 텍스트 입력은 parse_readings_text()로 먼저 변환하세요."

    try:
        raw_readings = list(readings)
    except TypeError:
        return False, "측정값은 반복 가능한 숫자 목록이어야 합니다."

    if len(raw_readings) == 0:
        return False, "데이터 없음: 측정값 목록이 비어 있습니다."

    cleaned_readings: list[float] = []
    for idx, value in enumerate(raw_readings, start=1):
        ok, number = _coerce_finite_float(value, f"측정값 {idx}번")
        if not ok:
            return False, str(number)

        if not (REBOUND_READING_MIN <= float(number) <= REBOUND_READING_MAX):
            return (
                False,
                f"측정값 {idx}번({float(number):g})이 허용 범위 "
                f"{REBOUND_READING_MIN:g}~{REBOUND_READING_MAX:g}를 벗어났습니다."
            )

        cleaned_readings.append(float(number))

    ok, angle_num = _coerce_finite_float(angle, "타격각도")
    if not ok:
        return False, str(angle_num)
    if not float(angle_num).is_integer():
        return False, f"타격각도는 정수 각도여야 합니다. 입력값: {float(angle_num):g}"
    angle_int = int(angle_num)
    if angle_int not in ALLOWED_REBOUND_ANGLES:
        allowed = ", ".join(f"{a}°" for a in sorted(ALLOWED_REBOUND_ANGLES))
        return False, f"허용되지 않는 타격각도입니다: {angle_int}°. 허용값: {allowed}"

    ok, days_num = _coerce_finite_float(days, "재령")
    if not ok:
        return False, str(days_num)
    if float(days_num) <= 0:
        return False, "재령은 0보다 큰 값이어야 합니다."

    ok, fck_num = _coerce_finite_float(design_fck, "설계강도")
    if not ok:
        return False, str(fck_num)
    if float(fck_num) <= 0:
        return False, "설계강도는 0보다 큰 값이어야 합니다."

    ok, ct_num = _coerce_finite_float(core_coeff, "코어 보정계수(Ct)")
    if not ok:
        return False, str(ct_num)
    if float(ct_num) <= 0:
        return False, "코어 보정계수(Ct)는 0보다 커야 합니다."

    if selected_formulas is None:
        normalized_formulas = None
    else:
        if isinstance(selected_formulas, str):
            formulas = [selected_formulas]
        else:
            try:
                formulas = list(selected_formulas)
            except TypeError:
                return False, "평균 산정 공식 선택값은 리스트 형태여야 합니다."

        normalized_formulas: list[str] = []
        invalid_formulas: list[str] = []
        for formula in formulas:
            name = str(formula).strip()
            if not name:
                continue
            if name not in REBOUND_FORMULA_NAMES:
                invalid_formulas.append(name)
                continue
            if name not in normalized_formulas:
                normalized_formulas.append(name)

        if invalid_formulas:
            allowed = ", ".join(REBOUND_FORMULA_OPTIONS)
            return False, f"알 수 없는 평균 산정 공식이 포함되어 있습니다: {invalid_formulas}. 허용값: {allowed}"

    return True, {
        "readings": cleaned_readings,
        "angle": angle_int,
        "days": float(days_num),
        "design_fck": float(fck_num),
        "core_coeff": float(ct_num),
        "selected_formulas": normalized_formulas,
    }


# =========================================================
# 반발경도 계산
# =========================================================
def get_angle_correction(R_val: Any, angle: Any) -> float:
    """
    엑셀 '타격방향 보정(ΔR)'과 동일한 2차식 적용.
    ΔR = a*R^2 + b*R + c
    """
    try:
        angle = int(angle)
        R = float(R_val)
    except (TypeError, ValueError):
        return 0.0

    if angle == 90:
        return (-0.0018 * R * R) + (0.2455 * R) - 11.906
    if angle == 45:
        return (-0.0026 * R * R) + (0.2563 * R) - 9.24
    if angle == -90:
        return (-0.0009 * R * R) + (0.0094 * R) + 4.48
    if angle == -45:
        return (-0.0007 * R * R) + (0.0129 * R) + 3.14
    return 0.0


def get_age_coefficient(days: Any) -> float:
    try:
        days = float(days)
    except (TypeError, ValueError):
        days = 3000.0

    age_table = {
        10: 1.55, 20: 1.12, 28: 1.00, 50: 0.87, 100: 0.78, 150: 0.74,
        200: 0.72, 300: 0.70, 500: 0.67, 1000: 0.65, 3000: 0.63,
    }
    sorted_days = sorted(age_table.keys())

    if days >= sorted_days[-1]:
        return age_table[sorted_days[-1]]
    if days <= sorted_days[0]:
        return age_table[sorted_days[0]]

    for i in range(len(sorted_days) - 1):
        d1, d2 = sorted_days[i], sorted_days[i + 1]
        if d1 <= days <= d2:
            c1, c2 = age_table[d1], age_table[d2]
            return c1 + (days - d1) / (d2 - d1) * (c2 - c1)

    return 1.0


def calculate_strength(
    readings: Iterable[Any] | None,
    angle: Any,
    days: Any,
    design_fck: Any = 24.0,
    selected_formulas: Iterable[str] | str | None = None,
    core_coeff: Any = 1.0,
    require_20_points: bool = True,
    point_count_policy: Any = None,
) -> tuple[bool, dict[str, Any] | str]:
    """
    반발경도 측정값으로 추정 압축강도를 계산합니다.

    반환:
    - 성공: (True, result_dict)
    - 실패: (False, error_message)
    """
    valid_input, validated = validate_rebound_inputs(
        readings=readings,
        angle=angle,
        days=days,
        design_fck=design_fck,
        selected_formulas=selected_formulas,
        core_coeff=core_coeff,
    )
    if not valid_input:
        return False, str(validated)

    assert isinstance(validated, dict)
    rd = validated["readings"]
    angle = validated["angle"]
    days = validated["days"]
    design_fck = validated["design_fck"]
    selected_formulas = validated["selected_formulas"]
    ct = validated["core_coeff"]

    n = len(rd)

    try:
        point_policy = normalize_rebound_point_policy(point_count_policy, require_20_points=require_20_points)
    except ValueError as e:
        return False, str(e)

    point_policy_label = get_rebound_point_policy_label(point_policy)
    point_policy_description = get_rebound_point_policy_description(point_policy)

    if point_policy == REBOUND_POINT_POLICY_EXACT_20 and n != 20:
        return False, (
            f"시험 무효: 측정점수 {n}개 입력됨. "
            "현재 정책은 '정확히 20개'이므로 1개소당 측정값을 정확히 20개로 입력하세요."
        )

    if point_policy == REBOUND_POINT_POLICY_MIN_20 and n < 20:
        return False, (
            f"시험 무효: 측정점수 {n}개 입력됨. "
            "현재 정책은 '20개 이상 허용'이므로 최소 20개 이상 입력해야 합니다."
        )

    if point_policy == REBOUND_POINT_POLICY_NO_MINIMUM and n < 1:
        return False, "시험 무효: 측정값이 없습니다."

    avg1 = float(np.mean(rd))

    low, high = avg1 * 0.8, avg1 * 1.2
    boundary_tol = 1e-12
    valid_mask = [(low - boundary_tol <= r <= high + boundary_tol) for r in rd]
    valid = [r for r, m in zip(rd, valid_mask) if m]
    excluded = [r for r, m in zip(rd, valid_mask) if not m]

    discard_ratio = (len(excluded) / n) if n else 1.0
    discard_limit = get_discard_limit_for_policy(n, point_policy)
    if point_policy == REBOUND_POINT_POLICY_EXACT_20:
        discard_rule = f"기각 {REBOUND_DISCARD_COUNT_LIMIT_20}개 이상"
    elif point_policy == REBOUND_POINT_POLICY_MIN_20:
        discard_rule = f"기각률 {REBOUND_DISCARD_RATIO_LIMIT * 100:.0f}% 이상(현재 {discard_limit}개 이상)"
    else:
        discard_rule = f"기각률 {REBOUND_DISCARD_RATIO_LIMIT * 100:.0f}% 이상(현재 {discard_limit}개 이상, 내부용)"

    if len(excluded) >= discard_limit:
        return False, (
            f"시험 무효: 기각 {len(excluded)}개({discard_ratio*100:.1f}%) → 재시험 권장 "
            f"(측정점수 정책: {point_policy_label}, 기각 기준: {discard_rule})"
        )

    if len(valid) == 0:
        return False, "유효 데이터 없음 (±20% 범위 내 값이 없습니다)"

    R_avg = float(np.mean(valid))
    corr = float(get_angle_correction(R_avg, angle))
    R0 = R_avg + corr
    age_c = float(get_age_coefficient(days))

    f_jsms = max(0.0, (1.27 * R0 - 18.0) * age_c)
    f_aij = max(0.0, (7.3 * R0 + 100.0) * 0.098 * age_c)
    f_mst = max(0.0, (15.2 * R0 - 112.8) * 0.098 * age_c)
    f_kwon = max(0.0, (2.304 * R0 - 38.80) * age_c)
    f_kalis = max(0.0, (1.3343 * R0 + 8.1977) * age_c)

    all_formulas_raw = {
        "일본재료": f_jsms,
        "일본건축": f_aij,
        "과기부": f_mst,
        "권영웅": f_kwon,
        "KALIS": f_kalis,
    }
    all_formulas = {k: v * ct for k, v in all_formulas_raw.items()}

    recommended_formulas = get_recommended_formulas(design_fck)
    if selected_formulas is None:
        formula_mode = "자동추천"
        applied_formulas = recommended_formulas
    else:
        formula_mode = "수동선택"
        applied_formulas = list(selected_formulas)
        if not applied_formulas:
            return False, "수동 공식 선택 모드에서는 평균 산정에 사용할 공식을 1개 이상 선택하세요."

    target_fs = [all_formulas[k] for k in applied_formulas if k in all_formulas]
    if not target_fs:
        return False, "평균 산정에 사용할 유효한 공식이 없습니다."

    s_mean = float(np.mean(target_fs))

    return True, {
        "N": n,
        "Effective_N": len(valid),
        "R_initial": avg1,
        "R_avg": R_avg,
        "Angle_Corr": corr,
        "R0": R0,
        "Age_Coeff": age_c,
        "Core_Coeff": ct,
        "Design_Fck": design_fck,
        "Angle": angle,
        "Days": days,
        "Point_Count_Policy": point_policy,
        "Point_Count_Policy_Label": point_policy_label,
        "Point_Count_Policy_Description": point_policy_description,
        "Discard": len(excluded),
        "Discard_Ratio": discard_ratio,
        "Discard_Limit": discard_limit,
        "Discard_Rule": discard_rule,
        "Excluded": excluded,
        "Formulas": all_formulas,
        "Formulas_Raw": all_formulas_raw,
        "Recommended_Formulas": recommended_formulas,
        "Selected_Formulas": selected_formulas,
        "Applied_Formulas": applied_formulas,
        "Formula_Mode": formula_mode,
        "Formula_Selection_Mode": "수동 선택" if formula_mode == "수동선택" else "자동추천",
        "Formula_Recommendation_Basis": get_recommended_formula_description(design_fck),
        "Mean_Strength": s_mean,
    }


# =========================================================
# 앱 화면의 검증 버튼에서도 재사용하는 테스트 목록
# =========================================================
def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def run_validation_tests() -> list[tuple[str, bool, Any]]:
    """
    앱 내부 검증 버튼용 테스트 결과 목록을 반환합니다.
    pytest 테스트 파일도 같은 케이스를 더 세분화해서 검증합니다.
    """
    results: list[tuple[str, bool, Any]] = []

    parser_cases = [
        ("54 56 55", [54.0, 56.0, 55.0]),
        ("54,56,55", [54.0, 56.0, 55.0]),
        ("54, 56, 55", [54.0, 56.0, 55.0]),
        ("54.5 56.0 55.5", [54.5, 56.0, 55.5]),
        ("54\n56\n55", [54.0, 56.0, 55.0]),
        ("[54, 56, 55]", [54.0, 56.0, 55.0]),
    ]
    parser_details = {raw: parse_readings_text(raw) for raw, _ in parser_cases}
    parser_pass = all(parse_readings_text(raw) == expected for raw, expected in parser_cases)
    results.append(("TC0(입력 파서)", parser_pass, parser_details))

    readings_tc1 = [
        58.4, 57.0, 61.8, 61.2, 60.6,
        58.9, 59.9, 58.9, 58.2, 57.8,
        61.5, 60.1, 64.1, 57.9, 59.3,
        56.8, 57.1, 58.0, 58.4, 58.0,
    ]
    ok, res = calculate_strength(
        readings_tc1, angle=90, days=3000,
        design_fck=40.0, selected_formulas=["일본재료", "일본건축", "과기부"],
        core_coeff=1.0, require_20_points=True,
    )
    tc1_pass = (
        ok and isinstance(res, dict)
        and _close(res["R_avg"], 59.195, 1e-3)
        and _close(res["Angle_Corr"], -3.680913945, 1e-6)
        and _close(res["R0"], 55.514086055, 1e-6)
        and _close(res["Age_Coeff"], 0.63, 1e-12)
        and _close(res["Formulas"]["일본재료"], 33.0768202086, 1e-6)
        and _close(res["Formulas"]["일본건축"], 31.194309588372, 1e-6)
        and _close(res["Formulas"]["과기부"], 45.132810978528, 1e-6)
    )
    results.append(("TC1(엑셀 일치)", tc1_pass, res if ok else res))

    ok2, res2 = calculate_strength([50] * 18 + [10, 90], angle=0, days=3000, design_fck=24, core_coeff=1.0)
    results.append(("TC2(기각 2개, 무효X)", ok2 and isinstance(res2, dict) and res2["Discard"] == 2, res2))

    ok3, res3 = calculate_strength([50] * 15 + [10, 90, 10, 90, 10], angle=0, days=3000, design_fck=24, core_coeff=1.0)
    results.append(("TC3(기각 5개, 무효)", (not ok3) and "시험 무효" in str(res3), res3))

    ok4a, res4a = calculate_strength(readings_tc1, angle=90, days=3000, design_fck=40, core_coeff=1.0)
    ok4b, res4b = calculate_strength(readings_tc1, angle=90, days=3000, design_fck=40, core_coeff=1.10)
    tc4_pass = (
        ok4a and ok4b and isinstance(res4a, dict) and isinstance(res4b, dict)
        and _close(res4b["Formulas"]["과기부"], res4a["Formulas"]["과기부"] * 1.10, 1e-6)
    )
    results.append(("TC4(Ct 배율)", tc4_pass, {
        "MST@1.0": res4a["Formulas"]["과기부"] if isinstance(res4a, dict) else res4a,
        "MST@1.10": res4b["Formulas"]["과기부"] if isinstance(res4b, dict) else res4b,
    }))

    validation_cases = {
        "NaN 측정값 차단": calculate_strength([50] * 19 + [np.nan], angle=0, days=3000, design_fck=24, core_coeff=1.0),
        "inf 측정값 차단": calculate_strength([50] * 19 + [np.inf], angle=0, days=3000, design_fck=24, core_coeff=1.0),
        "허용 범위 밖 측정값 차단": calculate_strength([50] * 19 + [5], angle=0, days=3000, design_fck=24, core_coeff=1.0),
        "잘못된 각도 차단": calculate_strength([50] * 20, angle=30, days=3000, design_fck=24, core_coeff=1.0),
        "재령 0 차단": calculate_strength([50] * 20, angle=0, days=0, design_fck=24, core_coeff=1.0),
        "설계강도 0 차단": calculate_strength([50] * 20, angle=0, days=3000, design_fck=0, core_coeff=1.0),
        "Ct 0 차단": calculate_strength([50] * 20, angle=0, days=3000, design_fck=24, core_coeff=0),
        "알 수 없는 공식 차단": calculate_strength([50] * 20, angle=0, days=3000, design_fck=24, selected_formulas=["없는공식"], core_coeff=1.0),
    }
    tc5_pass = all(not ok for ok, _ in validation_cases.values())
    results.append(("TC5(입력값 검증)", tc5_pass, {name: detail for name, (ok, detail) in validation_cases.items()}))

    ok6a, res6a = calculate_strength([50] * 20, angle=0, days=3000, design_fck=24, selected_formulas=None, core_coeff=1.0)
    ok6b, res6b = calculate_strength([50] * 20, angle=0, days=3000, design_fck=40, selected_formulas=None, core_coeff=1.0)
    ok6c, res6c = calculate_strength([50] * 20, angle=0, days=3000, design_fck=24, selected_formulas=["KALIS"], core_coeff=1.0)
    ok6d, res6d = calculate_strength([50] * 20, angle=0, days=3000, design_fck=24, selected_formulas=[], core_coeff=1.0)
    tc6_pass = (
        ok6a and isinstance(res6a, dict) and res6a["Formula_Mode"] == "자동추천" and res6a["Applied_Formulas"] == ["일본건축", "일본재료"]
        and ok6b and isinstance(res6b, dict) and res6b["Formula_Mode"] == "자동추천" and res6b["Applied_Formulas"] == ["과기부", "권영웅", "KALIS"]
        and ok6c and isinstance(res6c, dict) and res6c["Formula_Mode"] == "수동선택" and res6c["Applied_Formulas"] == ["KALIS"]
        and (not ok6d) and "1개 이상" in str(res6d)
    )
    results.append(("TC6(공식 선택 UX)", tc6_pass, {
        "24MPa 자동추천": res6a.get("Applied_Formulas") if isinstance(res6a, dict) else res6a,
        "40MPa 자동추천": res6b.get("Applied_Formulas") if isinstance(res6b, dict) else res6b,
        "수동선택 KALIS": res6c.get("Applied_Formulas") if isinstance(res6c, dict) else res6c,
        "수동 미선택": res6d,
    }))

    ok7a, res7a = calculate_strength([50] * 20, angle=0, days=3000, design_fck=24, core_coeff=1.0, point_count_policy=REBOUND_POINT_POLICY_EXACT_20)
    ok7b, res7b = calculate_strength([50] * 19, angle=0, days=3000, design_fck=24, core_coeff=1.0, point_count_policy=REBOUND_POINT_POLICY_EXACT_20)
    ok7c, res7c = calculate_strength([50] * 21, angle=0, days=3000, design_fck=24, core_coeff=1.0, point_count_policy=REBOUND_POINT_POLICY_EXACT_20)
    ok7d, res7d = calculate_strength([50] * 21, angle=0, days=3000, design_fck=24, core_coeff=1.0, point_count_policy=REBOUND_POINT_POLICY_MIN_20)
    ok7e, res7e = calculate_strength([50] * 19 + [20] * 5, angle=0, days=3000, design_fck=24, core_coeff=1.0, point_count_policy=REBOUND_POINT_POLICY_MIN_20)
    ok7f, res7f = calculate_strength([50] * 18 + [20] * 6, angle=0, days=3000, design_fck=24, core_coeff=1.0, point_count_policy=REBOUND_POINT_POLICY_MIN_20)
    tc7_pass = (
        ok7a and isinstance(res7a, dict) and res7a["Point_Count_Policy"] == REBOUND_POINT_POLICY_EXACT_20
        and (not ok7b) and "정확히 20개" in str(res7b)
        and (not ok7c) and "정확히 20개" in str(res7c)
        and ok7d and isinstance(res7d, dict) and res7d["Point_Count_Policy"] == REBOUND_POINT_POLICY_MIN_20 and res7d["N"] == 21
        and ok7e and isinstance(res7e, dict) and res7e["Discard"] == 5 and res7e["Discard_Limit"] == 6
        and (not ok7f) and "기각" in str(res7f)
    )
    results.append(("TC7(측정점수 정책)", tc7_pass, {
        "정확히20_20개": res7a.get("Point_Count_Policy_Label") if isinstance(res7a, dict) else res7a,
        "정확히20_19개": res7b,
        "정확히20_21개": res7c,
        "20개이상_21개": {"N": res7d.get("N"), "정책": res7d.get("Point_Count_Policy_Label")} if isinstance(res7d, dict) else res7d,
        "20개이상_24개중5개기각": {"Discard": res7e.get("Discard"), "Discard_Limit": res7e.get("Discard_Limit")} if isinstance(res7e, dict) else res7e,
        "20개이상_24개중6개기각": res7f,
    }))

    return results
