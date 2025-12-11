import streamlit as st
import pandas as pd

# ==================== 1. 각도 보정 및 로직 함수 ====================

def get_angle_correction(R_val, angle):
    """
    [타격 방향 보정 로직]
    사진의 보정표를 기준으로 R값에 보정치를 '더하는' 방식입니다.
    (하향 타격은 보정치가 음수이므로 빼지게 되고, 상향은 양수이므로 더해집니다.)
    """
    
    # === [중요] 보정표 데이터 (사진의 수치와 동일하게 설정) ===
    # 예: -90도(하향)일 때 R값이 20이면 -3.2를 더함(즉, 뺌)
    correction_table = {
        # 하향 타격 (-90도, 바닥)
        -90: {20: +3.4, 30: +3.1, 40: +2.7, 50: +2.2, 60: +1.7},
        
        # 사하향 타격 (-45도)
        -45: {20: +2.5, 30: +2.3, 40: +2.0, 50: +1.6, 60: +1.3},
        
        # 수평 타격 (0도) -> 보정 없음
        0:   {20: 0.0,  30: 0.0,  40: 0.0,  50: 0.0,  60: 0.0},
        
        # 사상향 타격 (+45도)
        45:  {20: -3.5, 30: -3.1, 40: -2.6, 50: -2.1, 60: -1.6},
        
        # 상향 타격 (+90도, 천장)
        90:  {20: -5.4, 30: -4.7, 40: -3.9, 50: -3.1, 60: -2.3}
    }
    
    # 1. 해당 각도의 데이터 가져오기 (없으면 수평 0으로 가정)
    if angle not in correction_table:
        angle = 0
    
    data = correction_table[angle]
    sorted_keys = sorted(data.keys()) # [20, 30, 40, 50, 60]
    
    # 2. 보간법(Interpolation) 적용
    # R값이 표에 있는 구간(예: 35) 사이에 있을 때 정확한 보정치를 계산
    
    # 범위 밖 처리 (최소값 20 미만, 최대값 60 초과 시 끝값 적용)
    if R_val <= sorted_keys[0]: return data[sorted_keys[0]]
    if R_val >= sorted_keys[-1]: return data[sorted_keys[-1]]
    
    # 구간 찾기 및 선형 보간
    for i in range(len(sorted_keys) - 1):
        r1 = sorted_keys[i]
        r2 = sorted_keys[i+1]
        
        if r1 <= R_val <= r2:
            v1 = data[r1]
            v2 = data[r2]
            # 비례식을 이용한 계산
            ratio = (R_val - r1) / (r2 - r1)
            return v1 + ratio * (v2 - v1)
            
    return 0.0

def get_age_coefficient(days):
    """재령 보정 로직 (3000일:0.63, 1000일:0.65, 500일:0.67)"""
    if days >= 3000: return 0.63
    elif days >= 1000: return 0.65
    elif days >= 500: return 0.67
    
    # 500일 미만 보간용 데이터
    points = [(0, 1.4), (20, 1.15), (28, 1.0), (50, 0.87), (90, 0.80), (365, 0.70), (500, 0.67)]
    x, y = [p[0] for p in points], [p[1] for p in points]
    
    if days <= x[0]: return y[0]
    for i in range(len(x)-1):
        if x[i] <= days <= x[i+1]:
            ratio = (days - x[i]) / (x[i+1] - x[i])
            return y[i] + ratio * (y[i+1] - y[i])
    return 0.67

# ==================== 2. 화면 구성 (UI) ====================
st.set_page_config(page_title="콘크리트 강도 산정", page_icon="🏗️")

st.title("🏗️ 반발경도 강도 산정 프로그램")
st.markdown("### 20개의 측정값(Raw Data) 입력")

# 입력 폼 구성
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        # 각도 선택을 좀 더 직관적으로 변경
        angle_option = st.selectbox(
            "타격 방향 (각도)", 
            options=[0, -90, -45, 45, 90],
            format_func=lambda x: f"{x}° (수평)" if x==0 else (f"{x}° (하향/바닥)" if x<0 else f"+{x}° (상향/천장)")
        )
    with col2:
        days_input = st.number_input("재령 (일수)", min_value=1, value=1000, step=10)

    input_text = st.text_area(
        "측정값 입력 (띄어쓰기 또는 쉼표 구분)", 
        "54 56 55 53 58 55 54 55 52 57 55 56 54 55 59 42 55 56 54 55",
        height=100
    )

# 계산 버튼
if st.button("🚀 강도 산정하기", type="primary", use_container_width=True):
    try:
        # 1. 데이터 전처리
        clean_text = input_text.replace(',', ' ').replace('\n', ' ')
        readings = [float(x) for x in clean_text.split() if x.strip()]
        
        if len(readings) < 5:
            st.error("❗ 데이터가 너무 적습니다. 최소 5개 이상의 값을 입력해주세요.")
        else:
            # 2. 통계 처리 (이상치 제거)
            avg1 = sum(readings) / len(readings)
            lower, upper = avg1 * 0.8, avg1 * 1.2
            valid = [r for r in readings if lower <= r <= upper]
            
            if not valid:
                st.error("유효한 데이터가 없습니다. (모든 값이 평균의 ±20%를 벗어남)")
            else:
                # 3. 계산 수행
                R_final = sum(valid) / len(valid)                 # 측정 R값 (평균)
                angle_corr = get_angle_correction(R_final, angle_option) # 각도 보정치
                
                # [핵심 로직] R0 = R + 보정치
                # 하향(-90)일 경우 angle_corr이 음수이므로 자연스럽게 빼짐
                R0 = R_final + angle_corr 
                
                age_coeff = get_age_coefficient(days_input)       # 재령 보정계수
                
                # 4. 강도 추정 공식 적용 (요청하신 수식 반영)
                
                # (1) 일본건축학회 (일반강도)
                # 식: (7.3 * Ro + 100) * 0.098
                f_aij = (7.3 * R0 + 100) * 0.098 * age_coeff
                
                # (2) 일본재료학회 (일반강도)
                # 식: 1.27 * Ro - 18.0
                f_jsms = (1.27 * R0 - 18.0) * age_coeff
                
                # (3) 과학기술부 (고강도)
                # 식: (15.2 * Ro - 112.8) * 0.098
                f_mst = (15.2 * R0 - 112.8) * 0.098 * age_coeff
                
                # (4) 권영웅 (고강도)
                # 식: 2.304 * Ro - 38.80
                f_kwon = (2.304 * R0 - 38.80) * age_coeff
                
                # 5. 결과 출력
                st.success("✅ 산정 완료")
                
                # 결과 요약 카드
                c1, c2, c3 = st.columns(3)
                c1.metric("측정 R값 (평균)", f"{R_final:.1f}")
                c2.metric("타격 보정치", f"{angle_corr:+.1f}") # 부호 표시 (+, -)
                c3.metric("최종 R0", f"{R0:.1f}", delta_color="normal")
                
                st.caption(f"※ 계산식: R0({R0:.1f}) = R({R_final:.1f}) + 보정치({angle_corr:.1f})")
                st.divider()

                # 강도 결과 테이블
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
                
                # 상세 정보
                with st.expander("📝 상세 정보 보기"):
                    st.write(f"- 유효 데이터: {len(valid)}개 / 입력 데이터: {len(readings)}개")
                    st.write(f"- 1차 평균(이상치 제거 전): {avg1:.2f}")
                    st.write(f"- 재령 일수: {days_input}일 (보정계수: {age_coeff:.3f})")

    except ValueError:
        st.error("⚠️ 입력값 오류: 숫자만 입력해주세요.")
