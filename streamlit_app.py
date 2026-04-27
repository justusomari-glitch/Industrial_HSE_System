import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import os
from dotenv import load_dotenv
import json


load_dotenv()
st.set_page_config(
    page_title="HEALTH AND SAFETY",
      page_icon=":factory:", 
      layout="wide",
      initial_sidebar_state="expanded"
      )
st.markdown("""
<style>
.main {background-color: #0E1117; color: white:}
.st.metric  {background-color:#1c1f26; padding: 15px; border-radius: 12px;}
.alert {padding:20px; border-radius: 10px; text-align: center; font-weight: bold;}
.critical {background-color: #ff4b4b;}
.warning {background-color: #ffa500;}
.safe {background-color: #28a745;}
</style>
""", unsafe_allow_html=True)

st.title("HEALTH AND SAFETY MANUAL INPUT & REAL-TIME MONITORING SYSTEM")
st.caption("AI-powered Industrial Risk Detection")

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
            st.write("API Response Status Code:", response.status_code)
            st.write("API Response Content:", response.text)
            result=response.json()
            st.write(result)
            if isinstance(result,list):
                    result=result[0]
            def to_float(val):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return 0.0
            incident_prob=to_float(result.get("incident_proba"))
            score=to_float(result.get("scores", 0.0))
            status=result.get("status","")
            severity=result.get("severity","")
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
            status=result.get("status","").title()
            risk_text=(result.get("reason") or "No specific reason provided.").capitalize()
                

            st.subheader("Safety Assesment")

            st.markdown(f"**Overal Status:** {status}")
            st.markdown(f"**Risk Insight:** {risk_text}")
                 
            st.subheader("Recommended Actions")
            st.warning(result.get("action").capitalize())
            st.subheader("Timeframe for Action")
            st.metric("Timeframe:", result.get("timeframe").capitalize())
            st.divider()
            llm_exp=result.get("llm_explanation","")
            if llm_exp:
                st.subheader("LLM Explanation")
                st.info(llm_exp)
            else:
                st.warning("No LLM explanation provided.")
            st.divider()
            st.subheader("SHAP EXPLANATION AND VISUALIZATION")
            shap_data=result.get("shap_explanation",{})
            if shap_data:
                st.subheader("SHAP Model Importance")
                shap_df=pd.DataFrame(
                list(shap_data.items()),
                columns=["Feature", "Impact"]
            ).sort_values("Impact",ascending=True)
            st.bar_chart(shap_df.set_index("Feature"))
            shap_sensor_data=result.get("shap_sensor_explanation",{})
            if shap_sensor_data:
                st.subheader("SHAP Sensor Impact on Anomaly Presence")
                sensor_df=pd.DataFrame({
                    "Sensor": list(shap_sensor_data.keys()),
                    "Impact": list(abs(v) for v in shap_sensor_data.values())
                }).sort_values("Impact",ascending=True)
                st.bar_chart(sensor_df.set_index("Sensor")['Impact'])
        except Exception as e:
                st.error(f"Error during prediction: {e}")
        

elif mode=="Real-time Monitoring":
    import time
    import pymysql
    import json

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
            cursor.execute("""
                            SELECT A.*,B.llm_explanation,B.timestamp as llm_timestamp,C.temperature_impact,C.humidity_impact,
                            C.noise_level_impact,C.gas_level_impact,C.vibration_impact,
                            C.voltage_impact,C.pressure_impact,C.co_ppm_impact,C.smoke_level_impact 
                            FROM healthandsafety as A 
                            LEFT JOIN llm_explanations as B ON A.id=B.healthandsafety_id
                            LEFT JOIN anomaly_shap_explanations as C ON A.id=C.healthandsafety_id
                            ORDER BY B.timestamp 
                            DESC LIMIT 1
                    """)
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
                    st.divider()
                    st.metric("Latest Anomaly:","Anomaly Detected" if latest.get("anomaly_binary")==1 else "No Anomaly")
                    st.divider()
                    col1,col2=st.columns(2)  
                    col1.metric("Latest Incident Type:",latest.get("incident_type"))
                    col2.metric("Latest Severity Type:",latest.get("severity"))
                    st.divider()
                    col1,col3 = st.columns(2)
                    col1.metric("Latest Incident Probability", f"{float(latest.get('incident_proba', 0.0)):.2f}")
                    col3.metric("Latest System Risk Score", f"{float(latest.get('scores', 0.0)):.2f}")
                    st.divider()
                    st.subheader("Latest Decision Engine Output")
                    shap_raw=latest.get("shap_explanation","{}")
                    shap_data=json.loads(shap_raw) if isinstance(shap_raw, str) else {}
                    if shap_data:
                        st.subheader("SHAP Feature Importance")
                        shap_df=pd.DataFrame({
                            "Feature": list(shap_data.keys()),
                            "Impact": list(shap_data.values())
                        }).sort_values("Impact",ascending=True)
                        st.bar_chart(shap_df.set_index("Feature")["Impact"])
                    shap_sensor={
                        "temperature_impact": abs(latest.get("temperature_impact")),
                        "humidity_impact": abs(latest.get("humidity_impact")),
                        "noise_level_impact": abs(latest.get("noise_level_impact")),
                        "gas_level_impact": abs(latest.get("gas_level_impact")),
                        "vibration_impact": abs(latest.get("vibration_impact")),
                        "voltage_impact": abs(latest.get("voltage_impact")),
                        "pressure_impact": abs(latest.get("pressure_impact")),
                        "co_ppm_impact": abs(latest.get("co_ppm_impact")),
                        "smoke_level_impact": abs(latest.get("smoke_level_impact")),
                    }
                    if any(shap_sensor.values()):
                        st.subheader("Sensor Anomaly Drivers")
                        sensor_df=pd.DataFrame({
                            "Sensor": list(shap_sensor.keys()),
                            "Impact": list(shap_sensor.values())
                        }).sort_values("Impact",ascending=False)
                        st.bar_chart(sensor_df.set_index("Sensor")["Impact"])
                    st.subheader("Latest Safety Assessment")
                    st.write(f"**Current Status:** {latest.get('status')}")
                    st.write(f"**Recommended action to Take:** {latest.get('action')}")
                    st.write(f"**Latest Reason:** {latest.get('reason')}")
                    st.write(f"**Window For Action:** {latest.get('timeframe')}")
                    llm_exp=latest.get("llm_explanation","")
                    if llm_exp:
                        st.subheader("LLM Explanation")
                        st.info(llm_exp)
                    st.markdown(f"### Last Updated: {latest.get('llm_timestamp')}")
        except Exception as e:
                st.error(f"Data base error: {e}")
        time.sleep(10)

        
    