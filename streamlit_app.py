import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import os
from dotenv import load_dotenv


load_dotenv()
st.set_page_config(
    page_title="HEALTH AND SAFETY",
      page_icon=":factory:", 
      layout="wide",
      initial_sidebar_state="expanded"
      )

st.title("HEALTH AND SAFETY")
st.caption("Health and Safety System")

mode= st.sidebar.selectbox(
    "Select Mode",
    ["Manual input", "Real-time Monitoring"]
)

if mode=="Manual input":

    st.sidebar.header("Input Parameters")
    st.sidebar.subheader("Product Information")
    shift = st.sidebar.selectbox("Shift Hours", ["Morning", "Afternoon","Night"])
    zone = st.sidebar.selectbox("Working Zones",['Zone_B_Moderate', 'Zone_D_HighRisk', 'Zone_C_Dangerous','Zone_A_Safe'])
    st.sidebar.subheader("Operator")
    st.sidebar.subheader("Machine Parameters")
    temperature = st.sidebar.slider("Temperature (°C)", min_value=10.0, max_value=135.0, value=40.0)
    humidity = st.sidebar.slider("Humidity", min_value=10.0, max_value=100.0, value=0.5)
    noise_level = st.sidebar.slider("Noise Level (dB)", min_value=10.0, max_value=150.0, value=15.0)
    gas_level = st.sidebar.slider("Gas Level", min_value=9.0, max_value=900.0, value=50.0)
    vibration = st.sidebar.slider("Vibration (mm/s)", min_value=0.0, max_value=20.0, value=4.0)
    voltage = st.sidebar.slider("Voltage (V)", min_value=20.0, max_value=300.0, value=60.0)
    co_ppm = st.sidebar.slider("CO PPM (ppm))", min_value=0.0, max_value=300.0, value=1.0)
    smoke_level = st.sidebar.slider("Ssmoke Level (mg/m)", min_value=1.0, max_value=500.0, value=0.5)
    pressure = st.sidebar.slider("Pressure (pa)",min_value=6.0, max_value=200.0, value=4.0)
    st.sidebar.subheader("Human Working Parametres")
    hours_worked = st.sidebar.slider("Number Of Hours Worked (hrs)", min_value=1.0, max_value=15.0, value=8.0)
    days_consecutive = st.sidebar.slider("Consucutive Days Worked", min_value=2, max_value=7, value=3)
    ppe_compliance = st.sidebar.slider("PPE Complience ()", min_value=0.0, max_value=1.0, value=0.1)
    break_compliance = st.sidebar.checkbox("Break compliance")

    run=st.sidebar.button("Predict Defect")
    if run:
        data={
            "temperature":temperature,
            "humidity":humidity,
            "noise_level": noise_level,
            "gas_level":gas_level,
            "vibration":vibration,
            "voltage":voltage,
            "pressure":pressure,
            "co_ppm":co_ppm,
            "smoke_level":smoke_level,
            "hours_worked":hours_worked,
            "days_consecutive":days_consecutive,
            "ppe_compliance":ppe_compliance,
            "break_compliance": break_compliance,
            "shift":shift,
            "zone":zone
        }
        url=st.secrets["API_URL"]
        try:
            response=requests.post(url,json=data)
            if response.status_code==200:
                result=response.json()
                if isinstance(result, list):
                    result=result[0]
                def to_float(val):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return 0.0
                incident_prob=to_float(result.get("incident_proba"))
                score=to_float(result.get("scores"))
                st.divider()

                st.write("**Anomaly Status:**",result.get("anomaly_binary"))
                st.write("**Severity:**",result.get("severity"))
                st.write("**Type of Incident:**",result.get("incident_type"))

                st.divider()
                if result.get("anomaly_binary")=="ANOMALY DETECTED":
                    st.error("Anomaly Detected! Please check machine immediately.")
                else:
                    st.success("No Anomaly Detected. Machine is stable.")
                history= np.clip(np.random.normal(score, scale=0.1, size=20), 0, 1)
                df=pd.DataFrame({
                    "timestamp":pd.date_range(start="2024-01-01", periods=20, freq="H"),
                    "risk_score":history
                })

                st.divider()

                col2,col4=st.columns(2)
            
                col2.metric("Incident Probability", f"{incident_prob:.2f}")
                col4.metric("System Risk Score", f"{score:.2f}")

                st.divider()

                st.subheader("Decision Engine Output")
                status=result.get("rule_engine","").title()
                risk_text=result.get("score_engine","")
                risk_text=risk_text.capitalize()

                st.subheader("Safety Assesment")

                st.markdown(f"**Overal Status:** {status}")
                st.markdown(f"**Risk Insight:** {risk_text}")

                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
        

elif mode=="Real-time Monitoring":
    import time
    import pymysql

    st.title("Real-time Monitoring")
    st.info("This mode will automatically fetch the latest predictions from the system every 10 seconds.")
    placeholder=st.empty()
    while True:
        try:
            connection=pymysql.connect(
                    host=os.getenv("DB_HOST"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASSWORD"),
                    database=os.getenv("DB_NAME"),
                    port=int(os.getenv("DB_PORT")),
                    ssl={"ssl": {"mode": os.getenv("SSL_MODE")}}
            )
            cursor=connection.cursor()
            cursor.execute("SELECT * FROM healthandsafety ORDER BY id DESC LIMIT 100")
            row=cursor.fetchall()
            columns=[col[0] for col in cursor.description]
            df=pd.DataFrame(row, columns=columns)
            connection.close()
            if not df.empty:
                latest=df.iloc[0]
                with placeholder.container():
                    st.dataframe(df)
                    st.subheader("Latest Prediction")
                    st.dataframe(df.tail(1))

                    col1,col2,col3=st.columns(3)
                    col1.metric("Latest Anomaly:","Anomaly Detected" if latest.get("anomaly_binary")==1 else "No Anomaly")
                    col2.metric("Latest Incident Type:",latest.get("incident_type"))
                    col3.metric("Latest Severity Type:",latest.get("severity"))
                    st.divider()
                    col1,col3 = st.columns(2)
                    col1.metric("Latest Incident Probability", f"{float(latest.get('incident_proba', 0.0)):.2f}")
                    col3.metric("Latest System Risk Score", f"{float(latest.get('scores', 0.0)):.2f}")
                    st.divider()
                    st.subheader("Latest Decision Engine Output")
                    st.markdown(f"##### Latest  Decision: {latest.get('rule_engine')}")
                    st.markdown(f"##### Latest System Decision: {latest.get('score_engine')}")
        except Exception as e:
                st.error(f"Data base error: {e}")
        time.sleep(10)
st.caption("Built by Justus Omari Kwache| Powered by FastAPI | Deployed on Render & Streamlit")
        
    