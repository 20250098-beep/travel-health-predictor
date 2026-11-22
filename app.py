import streamlit as st
import pandas as pd
import joblib
from datetime import date, timedelta

st.set_page_config(
    page_title="여행자 건강 위험 안내",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ 여행자 건강 위험 안내")
st.write("여행지의 기상 조건과 뎅기열 발생 데이터를 분석합니다.")

# 학습된 모델 불러오기
try:
    저장정보 = joblib.load("여행자_건강위험_모델.pkl")
    모델 = 저장정보["model"]
    특징목록 = 저장정보["features"]
except Exception as e:
    st.error(f"오류 종류: {type(e).__name__}")
    st.error(f"상세 내용: {repr(e)}")
    st.write("저장소 파일 목록:", __import__("os").listdir("."))
    st.stop()


# 모델 학습에 사용한 국가 목록 추출
국가특징 = [f for f in 특징목록 if f.startswith("Country_")]
국가목록 = [f.replace("Country_", "", 1) for f in 국가특징]

if not 국가목록:
    st.error("모델에서 국가 정보를 찾을 수 없습니다.")
    st.stop()

st.header("1. 여행 정보 입력")

국가 = st.selectbox("여행할 국가", 국가목록)

여행기간 = st.date_input(
    "여행 기간",
    value=(date.today(), date.today() + timedelta(days=7))
)

st.header("2. 여행지 기상 조건 입력")
st.caption("현재 버전은 기상 예보를 자동으로 가져오지 않습니다. 여행지의 예상 기상 값을 입력해 주세요.")

기온 = st.number_input("평균 기온 (℃)", value=25.0)
최고기온 = st.number_input("최고 기온 (℃)", value=30.0)
최저기온 = st.number_input("최저 기온 (℃)", value=20.0)
강수량 = st.number_input("강수량 (mm)", min_value=0.0, value=50.0)
습도 = st.number_input("평균 습도 (%)", min_value=0.0, max_value=100.0, value=70.0)

st.header("3. 건강 특성 입력")

건강특성 = st.multiselect(
    "해당하는 항목을 선택하세요.",
    ["어린이", "고령자", "기저질환이 있음", "면역력이 저하된 상태"]
)

if st.button("여행지 건강 위험 분석하기", type="primary"):

    입력자료 = pd.DataFrame(
        [[0] * len(특징목록)],
        columns=특징목록
    )

    국가열 = "Country_" + 국가
    if 국가열 in 입력자료.columns:
        입력자료.loc[0, 국가열] = 1

    기상값 = {
        "temperature_2m_mean (°C)": 기온,
        "temperature_2m_max (°C)": 최고기온,
        "temperature_2m_min (°C)": 최저기온,
        "precipitation_sum (mm)": 강수량,
        "relative_humidity_2m_mean (%)": 습도
    }

    for 변수, 값 in 기상값.items():
        if 변수 in 입력자료.columns:
            입력자료.loc[0, 변수] = 값

    예측 = 모델.predict(입력자료)[0]

    st.subheader("분석 결과")

    if 예측 == 1:
        st.warning("뎅기열 발생 사례와 유사한 조건으로 분류되었습니다.")
        st.write("모기 노출을 줄이고 여행지의 감염병 정보를 확인하세요.")
    else:
        st.success("뎅기열 발생 사례와 유사하지 않은 조건으로 분류되었습니다.")
        st.write("분석 결과와 관계없이 모기 예방 수칙을 지켜 주세요.")

    if 건강특성:
        st.subheader("건강 특성에 따른 주의사항")
        st.write("선택한 건강 특성을 고려해 여행 중 건강 상태를 더욱 주의 깊게 살펴보세요.")
        st.write("발열, 심한 두통, 근육통 등 증상이 나타나면 의료기관을 방문하세요.")

    st.info(
        "이 결과는 과거 국가별 뎅기열 발생 및 기상 데이터로 학습한 "
        "분류 모델의 결과입니다. 개인의 질병 발생 확률이나 의학적 진단을 의미하지 않습니다."
    )
