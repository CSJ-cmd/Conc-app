import streamlit as st
import pandas as pd

# ==================== 1. 보정 로직 함수 ====================

def get_angle_correction(R_val, angle):
    """
    [타격 방향 보정]
    - 사진의 표와 동일하게 R값 구간별 보정치를 적용
    - 보정치를 더하는 방식 (하향은 음수라 빼지고, 상향은 양수라 더해짐)
    """
    correction_table = {
        -90: {20: +3.2, 30: +3.1, 40: +2.7, 50: +2.2, 60: +1.7}, # 하향
        -45: {20: +2.4, 30: +2.3, 40: +2.0, 50: +1.6, 60: +1.3}, # 사하향
        0:   {20: 0.0,  30: 0.0,  40: 0.0,  50: 0.0,  60: 0.0},  # 수평
        45:  {20: -3.5, 30: -3.1, 40: -2.0, 50: -2.7, 60: -1.6}, # 사상향
        90:  {20: -5.4, 30: -4.7, 40: -3.9, 50: -3.1, 60: -2.3}  # 상향
    }
    
    if angle not in correction_table: angle = 0
    data = correction_table[angle]
    sorted_keys = sorted(data.keys())
    
    # 범위 밖 처리
    if R_val <= sorted_keys[0]: return data[sorted_keys[0]]
    if R_val >= sorted_keys[-1]: return data[sorted_keys[-1]]
    
    # 선형 보간
    for i in range(len(sorted_keys) - 1):
        r1, r2 = sorted_keys[i], sorted_keys[i+1]
        if r1 <= R_val <= r2:
            v1, v2 = data[r1], data[r2]
            ratio = (R_val - r1) / (r2 - r1)
            return v1 + ratio * (v2 - v1)
    return 0.0

def get_age_coefficient(days):
    """
    [재령 보정계수]
    - 표준 곡선(일본재료학회) + 장기 재령(사용자 정의) 통합
    - 표에 있는 일수 사이값은 자동으로 비례 계산(보간)됩니다.
    """
    # === [수정 포인트] 재령 보정표 데이터 ===
    # (일수: 계수)
    age_table = {
        10: 1.55,   # 초기 재령
        20: 1.12,
        28: 1.00,   # 기준
        50: 0.87,
        100: 0.78,
        150: 0.74,
        200: 0.72,
        300: 0.70
        500: 0.67,
        1000: 0.65,
        3000: 0.63  # 사용자 정의 구간 시작
    }
    
    sorted_days = sorted(age_table.keys())
    
    # 1. 3000일 이상은 0.63으로 고정
    if days >= sorted_days[-1]:
        return age_table[sorted_days[-1]]
    
    # 2. 10일 미만은 1.25 (또는 그 이상) 적용
    if days <= sorted_days[0]:
        return age_table[sorted_days[0]]
    
    # 3. 구간별 선형 보간 (Interpolation)
    for i in range(len(sorted_days) - 1):
        d1 = sorted_days[i]
        d2 = sorted_days[i+1]
        
        if d1 <= days <= d2:
            c1 = age_table[d1]
            c2 = age_table[d2]
            # 비례식: (입력일 - 시작일) / (끝일 - 시작일)
            ratio = (days - d1) / (d2 - d1)
            return c1 + ratio * (c2 - c1)
            
    return 1.0

# ==================== 2. 화면 구성 (UI) ====================
st.set_page_config(page_title="콘크리트 강도 산정", page_icon="🏗️")

st.title("🏗️ 반발경도 강도 산정")
st.markdown("##### 20개의 측정값과 조건을 입력하세요.")

# 입력 폼
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        angle_option = st.selectbox(
            "타격 방향 (각도)", 
            options=[0, -90, -45, 45, 90],
            format_func=lambda x: f"{x}° (수평)" if x==0 else (f"{x}° (하향/바닥)" if x<0 else f"+{x}° (상향/천장)")
        )
    with col2:
        days_input = st.number_input("재령 (일수)", min_value=10, value=1000, step=10)

    input_text = st.text_area(
        "측정값(Raw Data) 20개 입력", 
        "54 56 55 53 58 55 54 55 52 57 55 56 54 55 59 42 55 56 54 55",
        height=80
    )

# 계산 버튼
if st.button("🚀 강도 산정하기", type="primary", use_container_width=True):
    try:
        # 데이터 전처리
        clean_text = input_text.replace(',', ' ').replace('\n', ' ')
        readings = [float(x) for x in clean_text.split() if x.strip()]
        
        if len(readings) < 5:
            st.error("❗ 데이터가 너무 적습니다.")
        else:
            # 1. 이상치 제거
            avg1 = sum(readings) / len(readings)
            lower, upper = avg1 * 0.8, avg1 * 1.2
            valid = [r for r in readings if lower <= r <= upper]
            
            if not valid:
                st.error("유효한 데이터가 없습니다.")
            else:
                # 2. 값 계산
                R_final = sum(valid) / len(valid)
                angle_corr = get_angle_correction(R_final, angle_option)
                
                # R0 = R + 보정치
                R0 = R_final + angle_corr 
                
                # 재령 보정계수 (표에서 가져오기)
                age_coeff = get_age_coefficient(days_input)
                
                # 3. 강도 산정 (요청 수식)
                f_aij = (7.3 * R0 + 100) * 0.098 * age_coeff       # 건축학회
                f_jsms = (1.27 * R0 - 18.0) * age_coeff            # 재료학회
                f_mst = (15.2 * R0 - 112.8) * 0.098 * age_coeff    # 과기부
                f_kwon = (2.304 * R0 - 38.80) * age_coeff          # 권영웅
                
                # 4. 결과 표시
                st.success("✅ 산정 완료")
                
                # 요약 정보
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("평균 R값", f"{R_final:.1f}")
                c2.metric("타격 보정", f"{angle_corr:+.1f}")
                c3.metric("최종 R0", f"{R0:.1f}")
                c4.metric("재령 계수", f"{age_coeff:.3f}")

                st.divider()

                # 결과 테이블
                st.subheader("📊 압축강도 산정 결과")
                result_data = {
                    "구분": ["일본건축학회 (일반)", "일본재료학회 (일반)", "과학기술부 (고강도)", "권영웅 (고강도)"],
                    "추정 강도 (MPa)": [
                        f"{max(0, f_aij):.2f}",
                        f"{max(0, f_jsms):.2f}",
                        f"{max(0, f_mst):.2f}",
                        f"{max(0, f_kwon):.2f}"
                    ],
                    "적용 수식": [
                        "(7.3×Ro + 100) × 0.098", 
                        "1.27×Ro - 18.0", 
                        "(15.2×Ro - 112.8) × 0.098", 
                        "2.304×Ro - 38.80"
                    ]
                }
                st.table(pd.DataFrame(result_data))
                
                # 적용 기준 확인용 (Expandable)
                with st.expander("ℹ️ 적용된 보정 기준표 확인하기"):
                    st.markdown("**1. 재령 보정 계수표 (입력값에 따라 보간 적용)**")
                    age_df = pd.DataFrame({
                        "재령일수": [10, 20, 28, 50, 90, 365, 500, 1000, 3000],
                        "보정계수": [1.25, 1.15, 1.00, 0.87, 0.80, 0.70, 0.67, 0.65, 0.63]
                    })
                    st.dataframe(age_df, hide_index=True, use_container_width=True)
                    
                    st.markdown("**2. 상세 산정 정보**")
                    st.write(f"- 유효 데이터: {len(valid)}개 (기각: {len(readings)-len(valid)}개)")
                    st.write(f"- 1차 평균: {avg1:.2f}")

    except ValueError:
        st.error("⚠️ 숫자만 입력해주세요.")
