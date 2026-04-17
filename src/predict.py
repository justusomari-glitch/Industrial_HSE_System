from src.schema import HealthAndSafety
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from src.logger import log_prediction,setup_mlflow
import dagshub

app=FastAPI()
@app.on_event('startup')
def startup_event():
    setup_mlflow()

## load models
models_loaded = False
def load_models():
    global anomaly_model,incident_model,incident_severity,incident_type_model,models_loaded
    if models_loaded:
        return
    anomaly_model=joblib.load("models/anomaly_detection.pkl")
    incident_model=joblib.load("models/incident_model.pkl")
    incident_severity=joblib.load("models/incident_severity_model.pkl")
    incident_type_model=joblib.load("models/incident_type_model.pkl")
    models_loaded = True

type_weights={
    "Chemical":0.8,
    "Electrical":0.75,
    "Fire":1.0,
    "Mechanical":0.6,
    "None":0.1
}

severity_weights={
    "Medium":0.8,
    "High":1.0,
    "Low":0.4,
    "None":0.1
}

@app.get("/")
def home():
    return {"Message":"Health and Safety System Is up and running"}

@app.post("/predict")
def predict(data:HealthAndSafety):
    input_dict=data.model_dump()
    df=pd.DataFrame([input_dict])
    anomaly_flag=anomaly_model.predict(df)
    incident_proba=incident_model.predict_proba(df)[:,1]
    severity=incident_severity.predict(df)
    incident_type=incident_type_model.predict(df)

    ## adding the mcmd to join the four models
    anomaly_binary=np.where(anomaly_flag==-1,1,0)
    severity_score=np.array([severity_weights[m] for m in severity])
    type_score=np.array([type_weights[y] for y in incident_type])
    criteria=np.column_stack([anomaly_binary,incident_proba,severity_score,type_score])
    weights=([0.1,0.5,0.3,0.1])
    scores=np.dot(criteria,weights)

    ## creating our decision engine
    def rule_engine(row):
        anomaly=row["anomaly_binary"]
        incident_proba=row["incident_proba"]
        severity=row["severity"]
        incident_type=row["incident_type"]
        if incident_proba > 0.7 and severity== 'High':
            return "CRITICAL INCIDENT!!Immediate Shutdown"
        elif incident_proba > 0.7 and severity== 'Medium':
            return "HIGH RISK INCIDENT!!Urgent Mitigation"
        elif 0.4<=incident_proba<=0.7  and severity== 'High':
            return "POTENTIAL SEVERE INCIDENT!! Prepare Intervention"
        elif 0.4<=incident_proba<=0.7  and severity== 'Medium':
            return "MODERATE INCIDENT RISK!! Inspect Immediately"
        elif anomaly==1:
            return "OPERATIONAL ANOMALY DETECTED!! Carry Out Inspection"
        else:
            return "NORMAL OPERATIONS"  
        
    def score_engine(row):
        scores=row['scores']
        if scores >=0.7:
            return "CRITICAL RISK LEVEL: System conditions indicate a likelihood of a severe incident."
        elif scores >=0.5:
            return "HIGH RISK LEVEL: Elevated Risk Detected. Conditions may lead to an incident if not addresed promptly"
        elif scores >=0.3:
            return "MODERATE RISK LEVEL: Some risk factors present. Monitoring and preventive action recommended"
        else:
            return "LOW RISK LEVEL :System and Employees operating within normal conditions. No immediate action Needed"
        
    machines=pd.DataFrame({
        "anomaly_binary":anomaly_binary,
        "incident_proba":incident_proba,
        "severity":severity,
        "incident_type":incident_type,
        "scores":scores
    })
    machines["anomaly_binary"]=machines["anomaly_binary"].apply(
            lambda x: "ANOMALY DETECTED" if x==1 else "OKAY")
    machines['rule_engine']=machines.apply(rule_engine,axis=1)
    machines['score_engine']=machines.apply(score_engine,axis=1)

    return machines.to_dict(orient='records')
    

