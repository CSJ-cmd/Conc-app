import streamlit as st
import math
import pandas as pd
import numpy as np
import io
import altair as alt
import re
import logging
from PIL import Image

logger = logging.getLogger(__name__)

# =========================================================
# 1. 페이지 기본 설정 및 스타일
# =========================================================
st.set_page_config(
    page_title="구조물 안전진단 통합 평가 Pro",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        overflow-x: auto;
        white-space: nowrap;
        scrollbar-width: none;
        padding-left: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 5px 15px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
        font-size: 14px;
    }
    div[data-testid="stExpander"] details > summary {
        list-style: none !important;
        display: flex !important;
        align-items: flex-start !important;
        padding: 10px !important;
        height: auto !important;
        min-height: 40px;
        border: 1px solid #f0f2f6;
        border-radius: 8px;
    }
    div[data-testid="stExpander"] details > summary::-webkit-details-marker { display: none !important; }
    div[data-testid="stExpander"] details > summary > svg {
        margin-right: 12px !important;
        margin-top: 3px !important;
        width: 18px !important;
        min-width: 18px !important;
        height: 18px !important;
        flex-shrink: 0 !important;
        display: block !important;
    }
    /* 일부 모바일 환경에서 아이콘 폰트가 텍스트(arrow_right)로 노출되는 현상 대응 */
    div[data-testid="stExpander"] details > summary [class*="material"],
    div[data-testid="stExpander"] details > summary [data-testid*="icon"],
    div[data-testid="stExpander"] details > summary [aria-hidden="true"] {
        display: none !important;
    }
    div[data-testid="stExpander"] details > summary {
        padding-left: 12px !important;
    }
    div[data-testid="stExpander"] details > summary p {
        font-size: 15px;
        font-weight: 600;
        margin: 0;
        line-height: 1.5;
        white-space: normal !important;
        word-break: keep-all;
    }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; word-break: break-all; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem !important; }
    .calc-box { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4; margin-bottom: 15px; }
    div[data-testid="stTable"] { overflow-x: auto; }

    /* 모바일에서 좌측 접힘 컨트롤(아이콘 텍스트 노출) 숨김 */
    @media (max-width: 768px) {
        [data-testid="collapsedControl"] { display: none !important; }
        [data-testid="stHeader"] { height: 0 !important; }
        .block-container { padding-top: 0.5rem !important; }
        div[data-testid="stExpander"] details > summary {
            padding-left: 10px !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# 세션 상태 초기화 (데이터 연동용)
if 'rebound_data' not in st.session_state:
    st.session_state['rebound_data'] = []


def is_mobile_client():
    """간단한 UA 기반 모바일/태블릿 판별"""
    try:
        ua = str(st.context.headers.get("user-agent", "")).lower()
    except Exception:
        ua = ""
    mobile_keys = ["android", "iphone", "ipad", "mobile", "tablet"]
    return any(k in ua for k in mobile_keys)

# =========================================================
# 2. 핵심 로직 및 함수 정의
# =========================================================

@st.cache_resource
def load_ocr_model():
    import easyocr
    return easyocr.Reader(['en'], gpu=False)


def _normalize_ocr_token(text):
    """OCR 오인식 문자를 숫자 파싱 친화적으로 정규화"""
    text = str(text)

    # 천단위 콤마(예: 1,234)는 제거
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)

    replacements = {
        'O': '0', 'o': '0',
        'I': '1', 'l': '1', '|': '1',
        ',': '.', ';': '.',   # 기존 ';' -> ':' 오타성 치환 수정
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.strip()


def _extract_numeric_candidates(ocr_result):
    """easyocr detail=1 결과에서 숫자 후보를 추출"""
    candidates = []
    for item in ocr_result:
        if len(item) < 3:
            continue
        bbox, raw_text, conf = item
        text = _normalize_ocr_token(str(raw_text))
        nums = re.findall(r'\d+(?:\.\d+)?', text)
        if not nums:
            continue

        ys = [pt[1] for pt in bbox]
        xs = [pt[0] for pt in bbox]
        y_center = float(sum(ys) / len(ys))
        x_center = float(sum(xs) / len(xs))
        h = float(max(ys) - min(ys)) if ys else 0.0

        for num in nums:
            try:
                val = float(num)
                candidates.append({
                    "value": val,
                    "x": x_center,
                    "y": y_center,
                    "h": h,
                    "conf": float(conf),
                })
            except Exception:
                continue
    return candidates


def _cluster_rows(candidates):
    """y 좌표 기반으로 OCR 숫자 후보를 행 단위로 군집화"""
    if not candidates:
        return []

    heights = [c["h"] for c in candidates if c["h"] > 0]
    row_tol = max(10.0, (float(np.median(heights)) * 0.75) if heights else 12.0)

    sorted_by_y = sorted(candidates, key=lambda c: c["y"])
    rows = []
    for c in sorted_by_y:
        if not rows:
            rows.append([c])
            continue

        last_row = rows[-1]
        last_y = float(np.mean([r["y"] for r in last_row]))
        if abs(c["y"] - last_y) <= row_tol:
            last_row.append(c)
        else:
            rows.append([c])

    for row in rows:
        row.sort(key=lambda c: c["x"])
    return rows


def _select_best_20_readings(ocr_result, target_count=20):
    """
    전표형(영수증형) 이미지에서 측정값 영역을 우선 추출하고
    목표 개수(target_count, 기본 20개)에 맞춰 숫자 목록을 반환
    """
    candidates = _extract_numeric_candidates(ocr_result)
    if not candidates:
        return []

    # 반발경도 범위 중심 후보 우선
    plausible = [c for c in candidates if 10 <= c["value"] <= 100]
    work = plausible if plausible else candidates

    rows = _cluster_rows(work)
    if not rows:
        return []

    # 측정치 블록은 하단에 밀집해 나타나는 경우가 많으므로, 하단의 다수 숫자 행 우선
    measurement_rows = [r for r in rows if len(r) >= 3]
    if measurement_rows:
        selected_rows = measurement_rows[-max(4, min(6, len(measurement_rows))):]
    else:
        selected_rows = rows

    ordered_values = [c["value"] for row in selected_rows for c in row]
    ordered_keys = {(c["x"], c["y"], c["value"]) for row in selected_rows for c in row}

    if len(ordered_values) >= target_count:
        return ordered_values[:target_count]

    # 부족하면 나머지 후보를 y/x 순서대로 보충
    remain = [
        c["value"]
        for c in sorted(work, key=lambda c: (c["y"], c["x"]))
        if (c["x"], c["y"], c["value"]) not in ordered_keys
    ]
    merged = []
    for v in ordered_values + remain:
        if len(merged) >= target_count:
            break
        merged.append(v)
    return merged


def _format_readings_for_text(values):
    formatted = []
    for v in values:
        if abs(v - round(v)) < 1e-6:
            formatted.append(str(int(round(v))))
        else:
            formatted.append(f"{v:.1f}")
    return " ".join(formatted)


def parse_readings_text(raw_text):
    """텍스트 입력에서 숫자만 안전하게 파싱"""
    if raw_text is None:
        return []
    normalized = _normalize_ocr_token(str(raw_text))
    tokens = re.findall(r'[-+]?\d+(?:\.\d+)?', normalized)
    vals = []
    for t in tokens:
        try:
            vals.append(float(t))
        except Exception:
            continue
    return vals


def _safe_num(v, default, cast=float):
    n = pd.to_numeric(v, errors="coerce")
    if pd.isna(n):
        return cast(default)
    return cast(n)


def extract_numbers_from_image(image_input, ocr_mode="정밀"):
    """
    OCR 전처리 강화 + 전표형 측정지에서 20개 측정값 자동 추출
    """
    try:
        import cv2

        if isinstance(image_input, Image.Image):
            image = image_input.copy()
        else:
            image = Image.open(image_input)

        # cv2 처리 안정화를 위해 모드 정규화
        if image.mode not in ("RGB", "RGBA", "L"):
            image = image.convert("RGB")

        max_width = 800
        if image.width > max_width:
            ratio = max_width / float(image.width)
            new_height = int((float(image.height) * float(ratio)))
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            image = image.resize((max_width, new_height), resample)

        image_np = np.array(image)

        # 이미지 채널 수에 맞게 안전 변환
        if image_np.ndim == 2:
            gray = image_np
        elif image_np.shape[2] == 4:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        blur = cv2.medianBlur(gray, 3)

        # 다양한 조건 대응을 위한 전처리 후보군
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        th_adapt = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, th_clahe_otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morph = cv2.morphologyEx(th_adapt, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

        if ocr_mode == "빠른":
            variants = [gray, th_adapt]
        else:
            variants = [gray, th_adapt, th_otsu, th_clahe_otsu, morph]

        reader = load_ocr_model()

        best_values = []
        best_score = -1e9

        for processed in variants:
            result_detail = reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                allowlist='0123456789.:- '
            )
            values = _select_best_20_readings(result_detail, target_count=20)

            # 목표 20개 충족, 값 범위, 평균 confidence를 종합 점수화
            score = len(values) * 5
            if len(values) >= 20:
                score += 100
            in_range = sum(1 for v in values if 10 <= v <= 100)
            score += in_range * 2

            confs = [float(item[2]) for item in result_detail if len(item) >= 3]
            if confs:
                score += float(np.mean(confs)) * 10

            if score > best_score:
                best_score = score
                best_values = values

        # 실패 시 기존 방식처럼 전체 숫자라도 최대한 반환
        if not best_values:
            fallback = reader.readtext(gray, detail=0, allowlist='0123456789. ')
            fallback_nums = []
            for token in fallback:
                nums = re.findall(r'\d+(?:\.\d+)?', _normalize_ocr_token(str(token)))
                for n in nums:
                    try:
                        fallback_nums.append(float(n))
                    except Exception:
                        pass
            best_values = fallback_nums[:20]

        return _format_readings_for_text(best_values)

    except Exception as e:
        logger.exception("OCR 처리 중 오류 발생: %s", e)
        return ""


# ---------------------------------------------------------
# [수정 1] 타격방향 보정(ΔR) : 엑셀(1. 원본) 2차식 그대로
# ---------------------------------------------------------
def get_angle_correction(R_val, angle):
    """
    엑셀(1. 원본) '타격방향 보정(ΔR)'과 동일한 2차식 적용.
    ΔR = a*R^2 + b*R + c
    """
    try:
        angle = int(angle)
        R = float(R_val)
    except (TypeError, ValueError):
        return 0.0

    if angle == 90:     # 상향 수직
        return (-0.0018 * R * R) + (0.2455 * R) - 11.906
    elif angle == 45:   # 상향 경사
        return (-0.0026 * R * R) + (0.2563 * R) - 9.24
    elif angle == -90:  # 하향 수직
        return (-0.0009 * R * R) + (0.0094 * R) + 4.48
    elif angle == -45:  # 하향 경사
        return (-0.0007 * R * R) + (0.0129 * R) + 3.14
    else:               # 0° 수평(또는 그 외)
        return 0.0


def get_age_coefficient(days):
    try:
        days = float(days)
    except (TypeError, ValueError):
        days = 3000.0

    # 엑셀과 동일 테이블(보간)
    age_table = {
        10: 1.55, 20: 1.12, 28: 1.00, 50: 0.87, 100: 0.78, 150: 0.74,
        200: 0.72, 300: 0.70, 500: 0.67, 1000: 0.65, 3000: 0.63
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


# ---------------------------------------------------------
# [수정 2,3,5] 20점 기준 + 마스크 기반 기각 + Ct 반영
# ---------------------------------------------------------
def calculate_strength(
    readings,
    angle,
    days,
    design_fck=24.0,
    selected_formulas=None,
    core_coeff=1.0,          # Ct
    require_20_points=True   # 최소 20점 강제
):
    if not readings:
        return False, "데이터 없음"

    # 숫자화/정리
    try:
        rd = [float(x) for x in readings]
    except (TypeError, ValueError):
        return False, "측정값에 숫자가 아닌 값이 포함되어 있습니다."

    n = len(rd)

    # 최소 20점 기준
    if require_20_points and n < 20:
        return False, f"시험 무효: 측정점수 {n}개 (지침상 1개소당 20점 이상 필요)"

    # 1차 평균
    avg1 = float(np.mean(rd))

    # ±20% 기각 (마스크 기반: 중복값 오류 방지)
    low, high = avg1 * 0.8, avg1 * 1.2
    valid_mask = [(low <= r <= high) for r in rd]
    valid = [r for r, m in zip(rd, valid_mask) if m]
    excluded = [r for r, m in zip(rd, valid_mask) if not m]

    # 기각 비율 20% 초과 시 무효
    discard_ratio = (len(excluded) / n) if n else 1.0
    if discard_ratio > 0.20:
        return False, f"시험 무효: 기각 {len(excluded)}개({discard_ratio*100:.1f}%) → 재시험 권장"

    if len(valid) == 0:
        return False, "유효 데이터 없음 (±20% 범위 내 값이 없습니다)"

    # 유효 평균
    R_avg = float(np.mean(valid))

    # 타격방향 보정(엑셀 2차식)
    corr = float(get_angle_correction(R_avg, angle))
    R0 = R_avg + corr

    # 재령 계수
    age_c = float(get_age_coefficient(days))

    # Ct
    try:
        ct = float(core_coeff)
    except (TypeError, ValueError):
        ct = 1.0
    if ct <= 0:
        return False, "코어 보정계수(Ct)는 0보다 커야 합니다."

    # 강도식 (원값)
    f_jsms = max(0.0, (1.27 * R0 - 18.0) * age_c)                # 일본재료학회
    f_aij = max(0.0, (7.3 * R0 + 100.0) * 0.098 * age_c)         # 일본건축학회
    f_mst = max(0.0, (15.2 * R0 - 112.8) * 0.098 * age_c)        # 과기부(고강도)
    f_kwon = max(0.0, (2.304 * R0 - 38.80) * age_c)              # 권영웅
    f_kalis = max(0.0, (1.3343 * R0 + 8.1977) * age_c)           # KALIS

    all_formulas_raw = {
        "일본재료": f_jsms,
        "일본건축": f_aij,
        "과기부": f_mst,
        "권영웅": f_kwon,
        "KALIS": f_kalis,
    }

    # Ct 반영
    all_formulas = {k: v * ct for k, v in all_formulas_raw.items()}

    # 평균 산정 공식 선택
    if selected_formulas:
        target_fs = [all_formulas[k] for k in selected_formulas if k in all_formulas]
    else:
        # 설계강도 기준 자동추천 로직
        target_fs = ([all_formulas["일본건축"], all_formulas["일본재료"]]
                     if design_fck < 40
                     else [all_formulas["과기부"], all_formulas["권영웅"], all_formulas["KALIS"]])

    s_mean = float(np.mean(target_fs)) if target_fs else 0.0

    return True, {
        "N": n,
        "R_initial": avg1,
        "R_avg": R_avg,
        "Angle_Corr": corr,
        "R0": R0,
        "Age_Coeff": age_c,
        "Core_Coeff": ct,
        "Discard": len(excluded),
        "Excluded": excluded,
        "Formulas": all_formulas,
        "Formulas_Raw": all_formulas_raw,
        "Mean_Strength": s_mean
    }


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8-sig')


def to_excel(df):
    output = io.BytesIO()
    last_err = None

    for engine in ("xlsxwriter", "openpyxl"):
        try:
            output.seek(0)
            output.truncate(0)
            with pd.ExcelWriter(output, engine=engine) as writer:
                df.to_excel(writer, index=False, sheet_name='Result')
            return output.getvalue()
        except ImportError as e:
            last_err = e
            continue

    raise RuntimeError("엑셀 저장 엔진(xlsxwriter/openpyxl)이 설치되어 있지 않습니다.") from last_err


# =========================================================
# 2-1) 검증용 테스트 케이스 (엑셀 값과 동일한 케이스 포함)
# =========================================================
def run_validation_tests():
    """
    테스트 케이스:
    - TC1: 첨부 엑셀(1. 원본 / 2. 정리)와 수치가 일치하는 대표 케이스(상부구조 S1, 바닥판, 90°, 3000일)
    - TC2: 20점 중 2개 outlier(±20% 밖) -> 기각 2개, 무효 아님
    - TC3: 20점 중 5개 outlier -> 시험 무효 처리 확인
    - TC4: Ct=1.10 적용 시 강도들이 1.10배 되는지 확인
    """
    results = []

    # ----- TC1: 엑셀 일치 케이스 -----
    readings_tc1 = [
        58.4, 57.0, 61.8, 61.2, 60.6,
        58.9, 59.9, 58.9, 58.2, 57.8,
        61.5, 60.1, 64.1, 57.9, 59.3,
        56.8, 57.1, 58.0, 58.4, 58.0
    ]
    ok, res = calculate_strength(
        readings_tc1, angle=90, days=3000,
        design_fck=40.0, selected_formulas=["일본재료", "일본건축", "과기부"],
        core_coeff=1.0, require_20_points=True
    )

    exp_Ravg = 59.195
    exp_dR = -3.680913945
    exp_R0 = 55.514086055
    exp_age = 0.63
    exp_jsms = 33.0768202086
    exp_aij = 31.194309588372
    exp_mst = 45.132810978528

    def close(a, b, tol=1e-6):
        return abs(a - b) <= tol

    tc1_pass = (
        ok and
        close(res["R_avg"], exp_Ravg, 1e-3) and
        close(res["Angle_Corr"], exp_dR, 1e-6) and
        close(res["R0"], exp_R0, 1e-6) and
        close(res["Age_Coeff"], exp_age, 1e-12) and
        close(res["Formulas"]["일본재료"], exp_jsms, 1e-6) and
        close(res["Formulas"]["일본건축"], exp_aij, 1e-6) and
        close(res["Formulas"]["과기부"], exp_mst, 1e-6)
    )
    results.append(("TC1(엑셀 일치)", tc1_pass, res if ok else res))

    # ----- TC2: outlier 2개 -> 기각 2개 -----
    base = [50] * 18 + [10, 90]
    ok2, res2 = calculate_strength(base, angle=0, days=3000, design_fck=24, core_coeff=1.0, require_20_points=True)
    tc2_pass = ok2 and (res2["Discard"] == 2)
    results.append(("TC2(기각 2개, 무효X)", tc2_pass, res2 if ok2 else res2))

    # ----- TC3: outlier 5개 -> 무효 -----
    base3 = [50] * 15 + [10, 90, 10, 90, 10]
    ok3, res3 = calculate_strength(base3, angle=0, days=3000, design_fck=24, core_coeff=1.0, require_20_points=True)
    tc3_pass = (not ok3) and ("시험 무효" in str(res3))
    results.append(("TC3(기각 5개, 무효)", tc3_pass, res3))

    # ----- TC4: Ct=1.10 배율 확인 -----
    ok4a, res4a = calculate_strength(readings_tc1, angle=90, days=3000, design_fck=40, core_coeff=1.0, require_20_points=True)
    ok4b, res4b = calculate_strength(readings_tc1, angle=90, days=3000, design_fck=40, core_coeff=1.10, require_20_points=True)
    tc4_pass = ok4a and ok4b and close(res4b["Formulas"]["과기부"], res4a["Formulas"]["과기부"] * 1.10, 1e-6)
    results.append(("TC4(Ct 배율)", tc4_pass, {"MST@1.0": res4a["Formulas"]["과기부"], "MST@1.10": res4b["Formulas"]["과기부"]}))

    return results


# =========================================================
# 3. 메인 UI 구성
# =========================================================

st.title("🏗️ 구조물 안전진단 통합 평가 Pro")

with st.sidebar:
    st.header("⚙️ 프로젝트 정보")
    p_name = st.text_input("프로젝트명", "OO시설물 정밀점검")
    st.divider()
    st.caption("시설물안전법 및 세부지침 준수")

tab1, tab2, tab3, tab4 = st.tabs(["📖 점검 매뉴얼", "🔨 반발경도", "🧪 탄산화", "📈 통계·비교"])

# ---------------------------------------------------------
# [Tab 1] 점검 매뉴얼 + 검증 테스트
# ---------------------------------------------------------
with tab1:
    st.subheader("💡 프로그램 사용 가이드")
    st.info("""
    **1. 반발경도 산정 시 설계기준강도를 입력해주세요.**
    * 설계기준강도를 바탕으로 압축강도 추정에 필요한 공식 적용 로직이 자동으로 변경됩니다.

    **2. 타격방향 보정 값을 매뉴얼을 참고해서 상향 타격인지 하향타격인지를 구분해서 선택해주세요.**

    **3. 재령 등 별도로 적용하지 않을 시 프로그램상에서 재령 3000일, 설계기준강도 24MPa가 적용됩니다.**

    **4. 코어 보정계수(Ct)가 있으면 입력하세요.**
    * 최종 강도 = Ct × 비파괴(반발경도) 강도

    **5. 통계ㆍ비교 탭 활용 안내**
    * 추정된 압축강도의 표준편차와 변동계수 등을 계산하여 해당 시설물에 가장 적합한 산정식을 확인하고 검토하기 위함입니다.
    """)

    st.divider()
    st.subheader("📋 시설물 안전점검·진단 세부지침 매뉴얼")

    with st.expander("1. 반발경도 시험 (Rebound Hardness Test) 상세 지침", expanded=False):
        st.markdown("""
        #### **✅ 개요 및 원리**
        * 콘크리트 표면을 슈미트 해머로 타격하여 반발되는 거리($R$)를 측정하고, 이와 압축강도 사이의 상관관계를 통해 비파괴 강도를 추정합니다.

        #### **✅ 측정 및 기각 룰**
        1. **타격 점수**: 1개소당 **20점 이상** 측정을 원칙으로 합니다.
        2. **이상치 기각**: 전체 측정값의 산술평균을 낸 후, 평균값에서 **±20%를 벗어나는 데이터는 무효**
        3. **시험 무효**: 기각된 데이터가 **5개 이상(20% 초과)**인 경우 재시험 권장
        """)
        m_df = pd.DataFrame({
            "구분": ["상향 수직 (+90°)", "상향 경사 (+45°)", "수평 타격 (0°)", "하향 경사 (-45°)", "하향 수직 (-90°)"],
            "대상 부재 예시": ["슬래브 하부 (천장)", "보 경사면", "벽체, 기둥 측면", "교대/교각 경사부", "슬래브 상면 (바닥)"]
        })
        st.table(m_df)
        st.info("※ 본 프로그램은 각도 선택 시 엑셀(1. 원본) 보정식(2차식)을 그대로 적용하여 $R_0$를 산정합니다.")

    with st.expander("2. 탄산화 깊이 측정 (Carbonation Test) 상세 지침", expanded=False):
        st.markdown("""
        #### **✅ 탄산화 속도 및 수명 산식**
        * **$C = A\\sqrt{t}$** ($C$: 깊이, $A$: 속도계수, $t$: 년수)

        #### **✅ 등급 판정 기준 (잔여 피복 두께 기반)**
        * **A (매우 양호)**: 잔여 피복 두께 30mm 이상
        * **B (양호)**: 잔여 피복 두께 10mm ~ 30mm 미만
        * **C (보통)**: 잔여 피복 두께 0mm ~ 10mm 미만
        * **D (불량)**: 탄산화 깊이가 철근 위치를 초과 (잔여 피복 < 0)
        """)

    with st.expander("🧪 검증용 테스트 케이스 실행(개발/검증)", expanded=False):
        st.caption("TC1은 첨부 엑셀과 동일한 입력(20점)으로 계산했을 때 Ravg, ΔR, Ro, 강도식 값이 일치하는지 확인합니다.")
        if st.button("테스트 실행", type="primary"):
            test_results = run_validation_tests()
            for name, passed, detail in test_results:
                if passed:
                    st.success(f"{name}: PASS")
                else:
                    st.error(f"{name}: FAIL")
                st.code(str(detail))

# ---------------------------------------------------------
# [Tab 2] 반발경도 평가
# ---------------------------------------------------------
with tab2:
    st.subheader("🔨 반발경도 정밀 강도 산정")

    mobile_client = is_mobile_client()
    if mobile_client:
        st.caption("📱 모바일/태블릿 최적화 모드")

    mode = st.radio("입력 방식", ["단일 지점 (카메라/파일)", "다중 지점 (엑셀 업로드)"], horizontal=True)

    if mode.startswith("단일"):
        with st.container(border=True):
            st.markdown("##### 📸 측정값 입력")

            ocr_mode = st.radio(
                "OCR 처리 모드",
                ["빠른", "정밀"],
                horizontal=not mobile_client,
                index=1,
                help="빠른: 처리속도 우선 / 정밀: 인식률 우선"
            )

            cam_mode = st.toggle("💻 웹캠(PC) 모드로 전환하기", value=False)

            img_file = None
            rot_val = 0

            if not cam_mode:
                st.caption("📱 모바일: '사진 촬영' 선택 시 **후면 카메라(고화질/자동초점)**가 실행됩니다.")
                img_file = st.file_uploader("촬영 버튼 또는 갤러리 선택", type=['png', 'jpg', 'jpeg', 'bmp'])
            else:
                st.caption("💡 PC/노트북 웹캠을 사용할 때 적합합니다.")
                img_file = st.camera_input("측정 기록표를 촬영하세요")

            if img_file:
                st.caption("이미지가 회전되어 보이면 아래 버튼으로 조정 후 [계산 실행]을 눌러주세요.")
                rot_val = st.radio("이미지 회전(반시계)", [0, 90, 180, 270], index=0, horizontal=True, key="img_rot")

            if img_file is not None:
                # 새 업로드/회전/모드 변경 시 이전 OCR 결과 제거 (stale 방지)
                try:
                    upload_sig = (getattr(img_file, "name", ""), getattr(img_file, "size", 0), rot_val, ocr_mode, cam_mode)
                except Exception:
                    upload_sig = (str(img_file), rot_val, ocr_mode, cam_mode)

                if st.session_state.get("ocr_upload_sig") != upload_sig:
                    st.session_state["ocr_upload_sig"] = upload_sig
                    st.session_state.pop("ocr_result", None)

                with st.spinner("이미지 처리 및 숫자 인식 중..."):
                    pil_image = Image.open(img_file)
                    if rot_val != 0:
                        pil_image = pil_image.rotate(rot_val, expand=True)

                    recognized_text = extract_numbers_from_image(pil_image, ocr_mode=ocr_mode)

                    if recognized_text:
                        st.session_state['ocr_result'] = recognized_text
                        ocr_vals = parse_readings_text(recognized_text)
                        st.success(f"인식 성공 ({len(ocr_vals)}개): {recognized_text}")
                        if len(ocr_vals) != 20:
                            st.warning("자동 인식값이 20개가 아닙니다. 아래 입력창에서 확인/수정 후 계산하세요.")
                    else:
                        st.session_state.pop("ocr_result", None)
                        st.warning("숫자를 인식하지 못했습니다. 직접 입력해주세요.")

            # ---- 입력 파라미터: 모바일은 단일 컬럼, 데스크톱은 4열 ----
            if mobile_client:
                angle = st.selectbox(
                    "타격 방향",
                    [90, 45, 0, -45, -90],
                    format_func=lambda x: {90: "+90°(상향수직)", 45: "+45°(상향경사)", 0: "0°(수평)", -45: "-45°(하향경사)", -90: "-90°(하향수직)"}[x]
                )
                days = st.number_input("재령(일)", 10, 10000, 3000)
                fck = st.number_input("설계강도(MPa)", 15.0, 100.0, 24.0)
                ct = st.number_input("코어 보정계수 Ct", 0.10, 2.00, 1.00, step=0.01)
            else:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    angle = st.selectbox(
                        "타격 방향",
                        [90, 45, 0, -45, -90],
                        format_func=lambda x: {90: "+90°(상향수직)", 45: "+45°(상향경사)", 0: "0°(수평)", -45: "-45°(하향경사)", -90: "-90°(하향수직)"}[x]
                    )
                with c2:
                    days = st.number_input("재령(일)", 10, 10000, 3000)
                with c3:
                    fck = st.number_input("설계강도(MPa)", 15.0, 100.0, 24.0)
                with c4:
                    ct = st.number_input("코어 보정계수 Ct", 0.10, 2.00, 1.00, step=0.01)

            # 공식 선택 옵션
            formula_opts = ["일본재료", "일본건축", "과기부", "권영웅", "KALIS"]
            default_sels = ["일본건축", "일본재료"] if fck < 40 else ["과기부", "권영웅", "KALIS"]
            selected_methods = st.multiselect("평균 산정 적용 공식 (미선택 시 설계강도 기준 자동추천)", formula_opts, default=default_sels)

            default_txt = "54 56 55 53 58 55 54 55 52 57 55 56 54 55 59 42 55 56 54 55"
            if 'ocr_result' in st.session_state:
                default_txt = st.session_state['ocr_result']

            txt = st.text_area("측정값 (자동 인식 결과 수정 가능)", value=default_txt, height=120 if mobile_client else 80)

            # OCR 결과를 20칸 편집 UI로 제공 (txt 기준 동기화)
            preview_vals = parse_readings_text(txt)
            base_vals = (preview_vals + [np.nan] * 20)[:20]
            grid_df = pd.DataFrame({
                "No": list(range(1, 21)),
                "측정값": base_vals
            })
            grid_key = f"ocr_20_grid_{abs(hash(txt)) % (10**8)}"
            edited_grid = st.data_editor(
                grid_df,
                column_config={
                    "No": st.column_config.NumberColumn("No", disabled=True),
                    "측정값": st.column_config.NumberColumn("측정값", min_value=0.0, max_value=100.0, step=0.1),
                },
                hide_index=True,
                use_container_width=True,
                key=grid_key
            )

            valid_grid_vals = [float(v) for v in edited_grid["측정값"].tolist() if not pd.isna(v)]
            if valid_grid_vals:
                txt = " ".join([str(int(v)) if abs(v - round(v)) < 1e-6 else f"{v:.1f}" for v in valid_grid_vals])

        if st.button("계산 실행", type="primary", use_container_width=True):
            rd = parse_readings_text(txt)
            ok, res = calculate_strength(
                rd, angle, days,
                design_fck=fck,
                selected_formulas=selected_methods,
                core_coeff=ct,
                require_20_points=True
            )
            if ok:
                st.success(f"평균 추정 압축강도(코어보정 반영): **{res['Mean_Strength']:.2f} MPa**")

                with st.container(border=True):
                    r1, r2, r3 = st.columns(3)
                    r1.metric("유효 평균 R", f"{res['R_avg']:.3f}")
                    r2.metric("각도 보정 ΔR(엑셀식)", f"{res['Angle_Corr']:+.6f}")
                    r3.metric("정점수/기각", f"{res['N']} / {res['Discard']}")

                    r4, r5, r6 = st.columns(3)
                    r4.metric("최종 R₀", f"{res['R0']:.6f}")
                    r5.metric("재령 계수 α", f"{res['Age_Coeff']:.2f}")
                    r6.metric("Ct", f"{res['Core_Coeff']:.2f}")

                # 데이터 연동 버튼
                if st.button("➕ 통계 분석 목록에 추가", key="add_to_stats"):
                    st.session_state['rebound_data'].append(res['Mean_Strength'])
                    st.success(f"통계 탭 목록에 {res['Mean_Strength']:.2f} MPa 추가 완료!")

                df_f = pd.DataFrame({"공식": list(res["Formulas"].keys()), "강도": list(res["Formulas"].values())})
                chart = alt.Chart(df_f).mark_bar().encode(
                    x=alt.X('공식', sort=None),
                    y='강도',
                    color=alt.condition(alt.datum.강도 >= fck, alt.value('#4D96FF'), alt.value('#FF6B6B'))
                ).properties(height=350)

                rule_chart = alt.Chart(pd.DataFrame({'y': [fck]})).mark_rule(color='red', strokeDash=[5, 3], size=2).encode(y='y')
                st.altair_chart(chart + rule_chart, use_container_width=True)
            else:
                st.error(res)

    else:
        # ---------------------------------------------------------
        # [수정 4] 배치(엑셀) 템플릿 + 파싱 + 계산에 Ct 반영
        # ---------------------------------------------------------
        st.info("💡 엑셀 업로드 시 아래 양식을 다운로드하여 작성해주세요. (Ct 컬럼 추가됨)")

        template_df = pd.DataFrame({
            "지점": ["S1-Deck", "S2-Deck"],
            "각도": [90, -90],
            "재령": [3000, 3000],
            "설계": [40, 40],
            "Ct": [1.00, 1.00],
            "데이터": [
                "58.4 57 61.8 61.2 60.6 58.9 59.9 58.9 58.2 57.8 61.5 60.1 64.1 57.9 59.3 56.8 57.1 58 58.4 58.0",
                "32 33 35 34 32 33 35 34 32 33 35 34 32 33 35 34 32 33 35 34"
            ]
        })

        try:
            template_excel = to_excel(template_df)
            st.download_button(
                label="📥 입력 양식(엑셀) 다운로드",
                data=template_excel,
                file_name='반발경도_입력양식_Ct포함.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except RuntimeError as e:
            st.error(str(e))

        uploaded_file = st.file_uploader("작성된 파일 업로드", type=["csv", "xlsx"])
        init_data = []

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_up = pd.read_csv(uploaded_file)
                else:
                    df_up = pd.read_excel(uploaded_file)

                for idx, row in df_up.iterrows():
                    try:
                        init_data.append({
                            "선택": True,
                            "지점": row.get("지점", f"P{idx+1}"),
                            "각도": _safe_num(row.get("각도", 0), 0, int),
                            "재령": _safe_num(row.get("재령", 3000), 3000, int),
                            "설계": _safe_num(row.get("설계", 24.0), 24.0, float),
                            "Ct": _safe_num(row.get("Ct", 1.0), 1.0, float),
                            "데이터": str(row.get("데이터", ""))
                        })
                    except Exception as row_err:
                        logger.warning("업로드 %s행 파싱 실패: %s", idx + 1, row_err)

            except ImportError:
                st.error("엑셀 파일을 읽으려면 'openpyxl' 라이브러리가 필요합니다.")
            except Exception as e:
                st.error(f"파일 파싱 실패: {e}")

        df_batch = pd.DataFrame(init_data) if init_data else pd.DataFrame(columns=["선택", "지점", "각도", "재령", "설계", "Ct", "데이터"])
        edited_df = st.data_editor(
            df_batch,
            column_config={
                "선택": st.column_config.CheckboxColumn("선택", default=True),
                "각도": st.column_config.SelectboxColumn("각도", options=[90, 45, 0, -45, -90], required=True),
                "재령": st.column_config.NumberColumn("재령", default=3000),
                "설계": st.column_config.NumberColumn("설계", default=24),
                "Ct": st.column_config.NumberColumn("Ct", default=1.00),
            },
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic"
        )

        if st.button("🚀 일괄 계산 실행", type="primary", use_container_width=True):
            batch_res = []
            for _, row in edited_df.iterrows():
                if not row.get("선택", True):
                    continue

                try:
                    rd_list = parse_readings_text(row.get("데이터", ""))
                    ang_v = row.get("각도", 0)
                    age_v = row.get("재령", 3000)
                    fck_v = row.get("설계", 24.0)
                    ct_v = row.get("Ct", 1.0)

                    ok, res = calculate_strength(
                        rd_list,
                        ang_v,
                        age_v,
                        design_fck=fck_v,
                        selected_formulas=None,      # 배치 모드는 자동 추천 로직
                        core_coeff=ct_v,
                        require_20_points=True
                    )

                    if ok:
                        data_entry = {
                            "지점": row.get("지점", "P"),
                            "설계": float(fck_v),
                            "Ct": float(ct_v),
                            "추정강도": round(res["Mean_Strength"], 2),
                            "강도비(%)": round((res["Mean_Strength"] / float(fck_v)) * 100, 1) if float(fck_v) != 0 else np.nan,
                            "유효평균R": round(res["R_avg"], 3),
                            "보정ΔR": round(res["Angle_Corr"], 6),
                            "보정R0": round(res["R0"], 6),
                            "재령계수": round(res["Age_Coeff"], 2),
                            "기각수": int(res["Discard"]),
                            "기각데이터": str(res["Excluded"])
                        }
                        for f_name, f_val in res["Formulas"].items():
                            data_entry[f_name] = round(f_val, 3)

                        batch_res.append(data_entry)
                    else:
                        batch_res.append({
                            "지점": row.get("지점", "P"),
                            "설계": float(fck_v),
                            "Ct": float(ct_v),
                            "추정강도": np.nan,
                            "강도비(%)": np.nan,
                            "유효평균R": np.nan,
                            "보정ΔR": np.nan,
                            "보정R0": np.nan,
                            "재령계수": np.nan,
                            "기각수": np.nan,
                            "기각데이터": "",
                            "오류": str(res)
                        })
                except Exception as e:
                    batch_res.append({
                        "지점": row.get("지점", "P"),
                        "오류": f"행 처리 실패: {e}"
                    })

            if batch_res:
                final_df = pd.DataFrame(batch_res)
                res_tab1, res_tab2 = st.tabs(["📋 요약", "🔍 세부 데이터"])
                with res_tab1:
                    cols = ["지점", "설계", "Ct", "추정강도", "강도비(%)"]
                    cols = [c for c in cols if c in final_df.columns]
                    st.dataframe(final_df[cols], use_container_width=True, hide_index=True)
                with res_tab2:
                    st.dataframe(final_df, use_container_width=True, hide_index=True)

                st.divider()
                st.subheader("💾 결과 저장")
                try:
                    excel_data = to_excel(final_df)
                    st.download_button(
                        label="📊 전체 결과 엑셀 다운로드",
                        data=excel_data,
                        file_name=f"{p_name}_반발경도_평가결과_Ct포함.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                except RuntimeError as e:
                    st.error(str(e))

# ---------------------------------------------------------
# [Tab 3] 탄산화 평가
# ---------------------------------------------------------
with tab3:
    st.subheader("🧪 탄산화 깊이 및 상세 분석")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            m_depth = st.number_input("측정 깊이(mm)", 0.0, 100.0, 12.0)
        with c2:
            d_cover = st.number_input("설계 피복(mm)", 10.0, 200.0, 40.0)
        with c3:
            a_years = st.number_input("경과 년수(년)", 1, 100, 20)

    if st.button("평가 실행", type="primary", key="btn_carb_run", use_container_width=True):
        rate_a = m_depth / math.sqrt(a_years) if a_years > 0 else 0
        rem = d_cover - m_depth
        total_life = (d_cover / rate_a) ** 2 if rate_a > 0 else 99.9
        res_life = total_life - a_years

        grade, color = ("A", "green") if rem >= 30 else (("B", "blue") if rem >= 10 else (("C", "orange") if rem >= 0 else ("D", "red")))

        st.markdown(f"### 결과: :{color}[{grade} 등급]")
        with st.container(border=True):
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("잔여 피복량", f"{rem:.1f} mm")
            cc2.metric("속도 계수 (A)", f"{rate_a:.3f}")
            cc3.metric("예측 잔여수명", f"{max(0, res_life):.1f} 년")
            st.info(f"**계산 근거:** $A = {m_depth} / \\sqrt{{{a_years}}} = {rate_a:.3f}$, 잔여수명 $T = ({d_cover}/{rate_a:.3f})^2 - {a_years} = {res_life:.1f}$년")

        y_steps = np.linspace(0, 100, 101)
        d_steps = rate_a * np.sqrt(y_steps)
        df_p = pd.DataFrame({'경과년수': y_steps, '탄산화깊이': d_steps})
        line = alt.Chart(df_p).mark_line(color='#1f77b4').encode(
            x=alt.X('경과년수', title='경과년수 (년)'),
            y=alt.Y('탄산화깊이', title='탄산화 깊이 (mm)')
        )
        rule = alt.Chart(pd.DataFrame({'y': [d_cover]})).mark_rule(color='red', strokeDash=[5, 5], size=2).encode(y='y')
        point = alt.Chart(pd.DataFrame({'x': [a_years], 'y': [m_depth]})).mark_point(color='orange', size=100, filled=True).encode(x='x', y='y')
        st.altair_chart(line + rule + point, use_container_width=True)

# ---------------------------------------------------------
# [Tab 4] 통계 및 비교 (세션 연동 적용)
# ---------------------------------------------------------
with tab4:
    st.subheader("📊 강도 통계 및 비교 분석")
    c1, c2 = st.columns([1, 2])
    with c1:
        st_fck = st.number_input("기준 설계강도(MPa)", 15.0, 100.0, 24.0, key="stat_fck")

    session_data_str = " ".join([f"{x:.1f}" for x in st.session_state['rebound_data']])
    default_stat_txt = session_data_str if session_data_str else "24.5 26.2 23.1 21.8 25.5 27.0"

    with c2:
        raw_txt = st.text_area("강도 데이터 목록 (반발경도 탭에서 추가된 데이터 포함)", default_stat_txt, height=68)

    parsed = parse_readings_text(raw_txt)

    if parsed:
        df_stat = pd.DataFrame({"순번": range(1, len(parsed) + 1), "추정강도": parsed, "적용공식": ["전체평균(추천)"] * len(parsed)})

        label_df = st.data_editor(
            df_stat,
            column_config={
                "순번": st.column_config.NumberColumn("No.", disabled=True),
                "적용공식": st.column_config.SelectboxColumn("공식 선택", options=["일본건축", "일본재료", "과기부", "권영웅", "KALIS", "전체평균(추천)"], required=True)
            },
            use_container_width=True,
            hide_index=True
        )

        if st.button("통계 분석 실행", type="primary", use_container_width=True):
            strength_series = pd.to_numeric(label_df["추정강도"], errors="coerce").dropna()
            data = sorted(strength_series.astype(float).tolist())

            current_formulas = set(label_df["적용공식"].dropna().astype(str).unique())
            recommended = set(["일본건축", "일본재료", "전체평균(추천)"] if st_fck < 40 else ["과기부", "권영웅", "KALIS", "전체평균(추천)"])

            if not current_formulas.issubset(recommended):
                st.warning(f"⚠️ 주의: 현재 설계강도({st_fck}MPa) 기준, 일부 선택된 공식은 적합하지 않을 수 있습니다.")

            if len(data) >= 2:
                avg_v = float(np.mean(data))
                std_v = float(np.std(data, ddof=1))
                cv_v = (std_v / avg_v * 100.0) if avg_v != 0 else np.nan

                with st.container(border=True):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("평균", f"{avg_v:.2f} MPa", delta=f"{(avg_v / st_fck * 100):.1f}%")
                    m2.metric("표준편차 (σ)", f"{std_v:.2f} MPa")
                    m3.metric("변동계수 (CV)", f"{cv_v:.1f}%" if np.isfinite(cv_v) else "N/A")

                chart = alt.Chart(pd.DataFrame({"번호": range(1, len(data) + 1), "강도": data})).mark_bar().encode(
                    x='번호:O',
                    y='강도:Q',
                    color=alt.condition(alt.datum.강도 >= st_fck, alt.value('#4D96FF'), alt.value('#FF6B6B'))
                )
                rule = alt.Chart(pd.DataFrame({'y': [st_fck]})).mark_rule(color='red', strokeDash=[5, 3], size=2).encode(y='y')

                st.altair_chart(chart + rule, use_container_width=True)
            else:
                st.warning("통계 분석을 위해서는 유효한 숫자 데이터가 최소 2개 필요합니다.")

