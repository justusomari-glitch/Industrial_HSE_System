import os
from groq import Groq

def get_llm_explanations(input_dict,machines):
    api_key=os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    groq_client=Groq(api_key=api_key)
    llm_prompt=f"""
    You are an expert industrial safety analyst. 
    Based on the following machine data and model outputs,
    provide insights and recommandations for a safety officer:
    Sensor Readings:
    - Temperature: {'temperature'}
    - Humidity: {'humidity'}
    - Noise Level: {'noise_level'}
    - Gas Level: {'gas_level'}
    - Vibration: {'vibration'}
    - Voltage: {'voltage'}
    - Pressure: {'pressure'}
    - CO PPM: {'co_ppm'}
    - Smoke Level: {'smoke_level'}
    Human Working Parameters:
    - Hours Worked: {'hours_worked'}
    - Days Consecutive: {'days_consecutive'}
    - PPE Compliance: {'ppe_compliance'}
    - Break Compliance: {'break_compliance'}
    Model Outputs:
    - status : {machines['status'].iloc[0]}
    - MCDM Score: {machines['final_score'].iloc[0]}
    - Anomaly : {machines['anomaly_binary'].iloc[0]}
    - Incident Probability: {round(float(machines['incident_proba'].iloc[0]), 2)}
    - Severity: {machines['severity'].iloc[0]}
    - Incident Type: {machines['incident_type'].iloc[0]}
    - Key Contributing Factors: {{shap_explanation}}
    - Reason for Risk Level: {machines['reason'].iloc[0]}
    - Action To Be taken : {machines['action'].iloc[0]}
    - Timeframe for Action: {machines['timeframe'].iloc[0]}
    - Sensor influence on anomaly: {{shap_sensor_explanation}}
    Respond in 3 sentences or less. Be direct and actionable.No bullet points.
    """
    chat=groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"system","content":"You are an expert industrial safety analyst."},
                  {"role":"user","content":llm_prompt}],
        max_tokens=500
    )
    llm_explanation=chat.choices[0].message.content
    return llm_explanation