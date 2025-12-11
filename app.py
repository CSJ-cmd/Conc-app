import streamlit as st
import pandas as pd

# ==================== 1. 계산 로직 (수정됨) ====================
def get_angle_correction(R_val, angle):
    """타격 각도 보정 로직"""
    corrections = {
        -90: ([20, 30, 40, 50, 60], [-3.2, -3.4, -3.7, -4.1, -4.3]),
        -45: ([20, 30, 40, 50, 60], [-2.4, -2.6, -2.9, -3.1, -3.3]),
        0:   ([20, 30, 40, 50, 60], [0.0, 0.0, 0.0, 0.0, 0.0]),
        45:  ([20, 30, 40, 50, 60], [2.4, 2.3, 2.0, 1.6, 1.3]),
        90:  ([20, 30, 40, 50, 60], [3.2, 3.1, 2.7, 2.2, 1.7])
    }
    # 각도가 없으면 수평(0) 처리
    if angle not in corrections: angle = 0
    x, y = corrections[angle]
    
    # 보간법 적용
    if R_val <= x[0]: return y[0]
    if R_val >= x[-1]: return y[-1]
    
    for i in range(len(x)-1):
        if x[i] <= R_val <= x[i+1]:
            ratio = (R_val - x[i]) / (x[i+1] - x[i])
            return y[i] + ratio * (y[i+1] - y[i])
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
st.info("20개의 Raw Data를 입력하면 4가지 기준의 강도를 산출합니다.")

# 입력 폼 구성
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        angle_option = st.selectbox(
            "타격 방향 (각도)", 
            options=[0, -90, -45, 45, 90],
            format_func=lambda x: f"{x}° (수평)" if x==0 else (f"{x}° (하향/바닥)" if x<0 else f"+{x}° (상향/천장)")
        )
    with col2:
        days_input = st.number_input("재령 (일수)", min_value=1, value=1000, step=10, help="콘크리트 타설 후 경과 일수")

    input_text = st.text_area(
        "측정값(Raw Data) 20개 입력 (띄어쓰기 또는 쉼표로 구분)", 
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
                # 3. 핵심 변수 계산
                R_final = sum(valid) / len(valid)                 # 최종 평균 R
                angle_corr = get_angle_correction(R_final, angle_option) # 각도 보정치
                R0 = R_final + angle_corr                         # 보정된 R0
                age_coeff = get_age_coefficient(days_input)       # 재령 보정계수
                
                # 4. 강도 추정 공식 적용 (요청하신 수정 수식)
                
                # (1) 일본건축학회 (일반강도)
                # Fc = (7.3*Ro + 100) * 0.098 * 재령계수
                f_aij = (7.3 * R0 + 100) * 0.098 * age_coeff
                
                # (2) 일본재료학회 (일반강도) - 단위변환(0.098) 이미 포함된 계수로 추정됨 (1.27)
                # Fc = (1.27*Ro - 18.0) * 재령계수
                f_jsms = (1.27 * R0 - 18.0) * age_coeff
                
                # (3) 과학기술부 (고강도)
                # Fc = (15.2*Ro - 112.8) * 0.098 * 재령계수
                f_mst = (15.2 * R0 - 112.8) * 0.098 * age_coeff
                
                # (4) 권영웅 (고강도)
                # Fc = (2.304*Ro - 38.80) * 재령계수
                f_kwon = (2.304 * R0 - 38.80) * age_coeff
                
                # 5. 결과 출력
                st.success("✅ 산정이 완료되었습니다.")
                
                # (1) 주요 변수 요약
                c1, c2, c3 = st.columns(3)
                c1.metric("최종 R값 (평균)", f"{R_final:.1f}")
                c2.metric("각도 보정치", f"{angle_corr:.1f}")
                c3.metric("보정된 R0", f"{R0:.1f}", delta_color="normal")
                
                st.caption(f"※ 재령 보정계수: {age_coeff:.3f} (재령 {days_input}일 기준)")
                st.divider()

                # (2) 강도 결과 테이블 생성
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
                
                df_result = pd.DataFrame(result_data)
                st.table(df_result)
                
                # (3) 상세 정보 펼치기
                with st.expander("📝 산정 상세 정보 보기"):
                    st.write(f"- 입력 데이터 개수: {len(readings)}개")
                    st.write(f"- 유효 데이터 개수: {len(valid)}개 (기각 {len(readings)-len(valid)}개)")
                    st.write(f"- 1차 평균값: {avg1:.2f}")
                    st.write("- 적용된 재령 계수 기준: 500일(0.67), 1000일(0.65), 3000일(0.63)")
                    st.write("※ 모든 강도 값에는 재령보정계수가 최종적으로 곱해졌습니다.")

    except ValueError:
        st.error("⚠️ 입력값 오류: 숫자만 입력해주세요.")
