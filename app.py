
import streamlit as st
import joblib
import os

st.set_page_config(page_title="모델 점검", page_icon="🔍")

st.title("여행자 건강 위험 모델 점검")

파일이름 = "여행자_건강위험_모델.pkl"

st.write("현재 실행 위치:", os.getcwd())
st.write("저장소 파일 목록:", os.listdir("."))

if not os.path.exists(파일이름):
    st.error("모델 파일을 찾을 수 없습니다.")
    st.stop()

try:
    저장정보 = joblib.load(파일이름)

    st.success("모델 파일을 읽었습니다.")
    st.write("저장된 자료의 종류:", type(저장정보).__name__)

    if isinstance(저장정보, dict):
        st.write("저장된 항목:", list(저장정보.keys()))

        if "model" in 저장정보:
            st.write("모델 종류:", type(저장정보["model"]).__name__)

        if "features" in 저장정보:
            st.write("특징 변수 개수:", len(저장정보["features"]))

    else:
        st.warning("저장된 자료가 딕셔너리 형식이 아닙니다.")

except Exception as e:
    st.error("모델 읽기 오류")
    st.exception(e)
