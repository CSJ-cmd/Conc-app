import streamlit as st
import pandas as pd

# ==================== 1. 계산 로직 ====================
def get_angle_correction(R_val, angle):
    corrections = {
        -90: ([20, 30, 40, 50, 60], [-3.2, -3.4, -3.7, -4.1, -4.3]),
        -45: ([20, 30, 40, 50, 60], [-2.4, -2.6, -2.9, -3.1, -3.3]),
        0:   ([20, 30, 40, 50, 60], [0.0, 0.0, 0.0, 0.0, 0.0]),
        45:  ([20, 30, 40, 50, 60], [2.4, 2.3, 2.0, 1.6, 1.3]),
        90:  ([20, 30, 40, 50, 60], [3.2, 3.1, 2.7, 2.2, 1.7])
    }
    if angle not in corrections: angle = 0
    x, y = corrections[angle]
    if R_val <= x[0]: return y[0]
    if R_val >= x[-1]: return y[-1]
    for i in range(len(x)-1):
        if x[i] <= R_val <= x[i+1]:
            ratio = (R_val - x[i]) / (x[i+1] - x[i])
            return y[i] + ratio * (y[i+1] - y[i])
    return 0.0

def get_age_coefficient(days):
    if days >= 3000: return 0.63
    elif days >= 1000: return 0.65
    elif days >= 500: return 0.67
    points = [(0, 1.4), (20, 1.15), (28, 1.0), (50, 0.87), (90, 0.80), (365, 0.70), (500, 0.67)]
    x, y = [p[0] for p in points], [p[1] for p in points]
    if days <= x[0]: return y[0]
    for i in range(len(x)-1):
        if x[i] <= days <= x[i+1]:
            ratio = (days - x[i]) / (x[i+1] - x[i])
            return y[i] + ratio * (y[i+1] - y[i])
    return 0.67

# ==================== 2. 화면 구성 (UI) ====================
st.set_page_config(page_title="콘크리트 강도 추정", page_icon="🔨")
st.title("🔨 반발경도 강도 추정기")

col1, col2 = st.columns(2)
with col1:
    angle_option = st.selectbox("타격 방향", [0, -90, -45, 45, 90], format_func=lambda x: "수평 (0°)" if x==0 else f"{x}°")
with col2:
    days_input = st.number_input("재령 (일수)", min_value=1, value=1000)

input_text = st.text_area("측정값 20개 입력", "54 56 55 53 58 55 54 55 52 57 55 56 54 55 59 42 55 56 54 55")

if st.button("🚀 강도 계산", type="primary"):
    try:
        readings = [float(x) for x in input_text.replace(',', ' ').split()]
        if len(readings) < 5: st.error("데이터 부족")
        else:
            avg1 = sum(readings)/len(readings)
            valid = [r for r in readings if avg1*0.8 <= r <= avg1*1.2]
            if not valid: st.error("유효값 없음")
            else:
                R_final = sum(valid)/len(valid)
                R0 = R_final + get_angle_correction(R_final, angle_option)
                age = get_age_coefficient(days_input)
                f_kepco = (14.29*R0 - 8.057)*0.1*age
                f_kwon = (2.304*R0 - 38.80)*age
                
                st.success("계산 완료")
                st.table(pd.DataFrame({
                    "구분": ["최종 R값", "보정 R0", "한전(MPa)", "권영웅(MPa)"],
                    "결과": [f"{R_final:.1f}", f"{R0:.1f}", f"{max(0,f_kepco):.2f}", f"{max(0,f_kwon):.2f}"]
                }))
    except: st.error("숫자만 입력하세요")
