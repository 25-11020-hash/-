import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 페이지 설정 ---
st.set_page_config(page_title="반도체 공정 AI 모니터링", layout="wide")

st.title("🔬 반도체 웨이퍼 불량 탐지 및 공정 모니터링 시스템")
st.markdown("**반도체 인공지능 공학과 포트폴리오** | 센서 데이터를 활용한 실시간 불량 예측 대시보드")
st.write("---")

# --- 1. 가상 데이터 생성 (반도체 센서 데이터 시뮬레이션) ---
@st.cache_data
def load_semiconductor_data():
    np.random.seed(42)
    num_samples = 200
    
    # 반도체 에칭/증착 공정 주요 파라미터 가정
    temperature = np.random.normal(loc=180, scale=10, size=num_samples) # 챔버 온도 (°C)
    pressure = np.random.normal(loc=50, scale=5, size=num_samples)       # 압력 (mTorr)
    gas_flow = np.random.normal(loc=120, scale=12, size=num_samples)     # 가스 유량 (sccm)
    
    data = pd.DataFrame({
        'Chamber_Temp': temperature,
        'Pressure': pressure,
        'Gas_Flow': gas_flow
    })
    
    # 특정 조건(예: 온도가 너무 높거나 압력이 너무 낮으면)일 때 불량(1) 발생으로 레이블링
    # 실제 공정의 이상 현상을 단순화한 로직입니다.
    def label_defect(row):
        if row['Chamber_Temp'] > 195 or row['Pressure'] < 42 or row['Gas_Flow'] > 140:
            return 1 # 불량 (Defect)
        else:
            return 0 # 정상 (Normal)
            
    data['Status'] = data.apply(label_defect, axis=1)
    return data

df = load_semiconductor_data()

# --- 2. 사이드바: 가상 AI 모델 학습 ---
st.sidebar.header("🤖 AI 모델 설정")
test_size = st.sidebar.slider("테스트 데이터 비율 (%)", 10, 50, 20, step=5) / 100

# 모델 학습
X = df[['Chamber_Temp', 'Pressure', 'Gas_Flow']]
y = df['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.sidebar.success(f"모델 학습 완료! 정확도(Accuracy): {acc*100:.1f}%")

# --- 3. 메인 화면: 실시간 공정 데이터 입력 및 예측 ---
st.header("📥 실시간 센서 데이터 입력 (테스트)")
col1, col2, col3 = st.columns(3)

with col1:
    input_temp = st.number_input("챔버 온도 (°C)", min_value=140.0, max_value=220.0, value=180.0)
with col2:
    input_press = st.number_input("챔버 압력 (mTorr)", min_value=30.0, max_value=70.0, value=50.0)
with col3:
    input_gas = st.number_input("가스 유량 (sccm)", min_value=80.0, max_value=160.0, value=120.0)

# 예측 버튼 클릭 시
if st.button("🚨 불량 여부 진단하기", use_container_width=True):
    input_data = np.array([[input_temp, input_press, input_gas]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.write("---")
    if prediction == 1:
        st.error(f"⚠️ **[위험] 불량(Defect) 웨이퍼 가능성이 높습니다!** (확률: {probability[1]*100:.1f}%)")
    else:
        st.success(f"✅ **[정상] 안정적인 공정 상태입니다.** (정상 확률: {probability[0]*100:.1f}%)")

st.write("---")

# --- 4. 데이터 시각화 ---
st.header("📊 전체 공정 데이터 분석 리포트")

tab1, tab2 = st.tabs(["데이터 테이블", "공정 산점도 분석"])

with tab1:
    st.dataframe(df, use_container_width=True)
    st.caption(f"총 데이터 수: {len(df)}개 | 정상: {len(df[df['Status']==0])}개 | 불량: {len(df[df['Status']==1])}개")

with tab2:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1번 그래프: 온도 vs 압력
    sns.scatterplot(data=df, x='Chamber_Temp', y='Pressure', hue='Status', palette={0: '#2ecc71', 1: '#e74c3c'}, ax=ax[0])
    ax[0].set_title("Chamber Temperature vs Pressure")
    ax[0].set_xlabel("Temperature (°C)")
    ax[0].set_ylabel("Pressure (mTorr)")
    
    # 2번 그래프: 온도 vs 가스 유량
    sns.scatterplot(data=df, x='Chamber_Temp', y='Gas_Flow', hue='Status', palette={0: '#2ecc71', 1: '#e74c3c'}, ax=ax[1])
    ax[1].set_title("Chamber Temperature vs Gas Flow")
    ax[1].set_xlabel("Temperature (°C)")
    ax[1].set_ylabel("Gas Flow (sccm)")
    
    st.pyplot(fig)
