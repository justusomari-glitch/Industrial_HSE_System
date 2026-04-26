from src.schema import HealthAndSafety
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from src.logger import log_prediction,setup_mlflow
import dagshub
import shap

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
    load_models()
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
    scores=np.dot(criteria,weights)[0]

    ## shap explainabilty
    feature_names=["anomaly_binary","incident_proba","severity_score","type_score"]
    background=np.zeros((1,4))
    explainer=shap.KernelExplainer(lambda x: np.dot(x,weights),background)
    shap_values=explainer.shap_values(criteria)
    if isinstance(shap_values, list):
        shap_values=shap_values[0]
    else:
        shap_values=shap_values
    shap_explanation={
        feature_names[i]: round(float(shap_values[0][i]), 2) 
        for i in range(len(feature_names))
    }

    ## creating our decision engine
    ## step one is that the score engine returns a number not text
    # define the risk levels
    RISK_LEVELS = {
        "LOW RISK!!": 1,
        "MODERATE RISK!!": 2,
        "HIGH RISK!!": 3,
        "CRITICAL RISK!!": 4
    }
    
    ## rule engine (This is split into 2)
    def apply_soft_rules(scores,row):
        if row['anomaly_binary'] == 1:
            scores +=0.05
        if row['severity'] == "High":
            scores +=0.05
        return min(scores,1.0)
    def check_hard_rules(row):
        if row['incident_proba']>0.8 and row['severity'] == "High":
            return "CRITICAL RISK!!","Critical probability and Severity."
        if row['anomaly_binary']== 1 and row['incident_proba']>0.7:
            return "HIGH RISK!!", "Anomaly Detected with High Incident Probability."
        return None,None
    
    def score_engine(row):
        return row['scores']
    
    ## decision engine
    def rule_engine(row):
        scores =score_engine(row)

        # apply the soft rules
        scores=apply_soft_rules(scores,row)

        # classify rules
        if scores >0.7:
            status = "CRITICAL RISK!!"
            default_reason="Critical risk identified based on combined factors."
        elif scores >0.5:
            status = "HIGH RISK!!"
            default_reason="High risk identified based on combined factors."
        elif scores >0.3:
            status = "MODERATE RISK!!"
            default_reason="Moderate risk identified based on combined factors."
        else:
            status = "LOW RISK!!"
            default_reason="Low risk identified based on combined factors."
        overide_status,reason=check_hard_rules(row)
        if overide_status:
            if RISK_LEVELS[overide_status]>RISK_LEVELS[status]:
                status=overide_status
                final_reason=reason
            else:
                final_reason=default_reason
        else:
            final_reason=default_reason
        return {
            "score":round(scores,2),
            "status":status,
            "reason":final_reason
        }
    # action mapping based on score
    def action_mapping(status):
        mapping={
            "CRITICAL RISK!!": ("Immediate Evacuation and Emergency Response Required.",
                                  "<24 hours"),
            "HIGH RISK!!": ("Urgent Intervention Needed. Address Issues within 24-48 hours.", 
                            "24-48 hours"),
            "MODERATE RISK!!": ("Monitor Closely and Implement Preventive Measures.",
                                 "48-72 hours"),
            "LOW RISK!!": ("Continue Regular Operations with Standard Safety Protocols.",
                            "No immediate action needed")
        }
        return mapping.get(status, ("UNKNOWN STATUS - INVESTIGATE IMMEDIATELY.",
                                    "IMMEDIATE"))
    machines=pd.DataFrame({
        "anomaly_binary":anomaly_binary,
        "incident_proba":incident_proba,
        "severity":severity,
        "incident_type":incident_type,
        "scores":scores
    })
    machines["anomaly_flag"]=machines["anomaly_binary"].apply(
            lambda x: "ANOMALY DETECTED" if x==1 else "OKAY")
    machines['decision']=machines.apply(rule_engine,axis=1)
    machines['status']=machines['decision'].apply(lambda x: x['status'])
    machines['reason']=machines['decision'].apply(lambda x: x['reason'])
    machines['final_score']=machines['decision'].apply(lambda x: x['score'])
    machines[['action', 'timeframe']]=machines['status'].apply(lambda x: pd.Series(action_mapping(x)))
    log_prediction(
        # __inputs__
        temperature=input_dict['temperature'],
        humidity=input_dict['humidity'],  
        noise_level=input_dict['noise_level'],
        gas_level=input_dict['gas_level'],
        vibration=input_dict['vibration'],
        voltage=input_dict['voltage'],
        pressure=input_dict['pressure'],
        co_ppm=input_dict['co_ppm'],
        smoke_level=input_dict['smoke_level'],
        hours_worked=input_dict['hours_worked'],
        days_consecutive=input_dict['days_consecutive'],
        ppe_compliance=input_dict['ppe_compliance'],
        break_compliance=input_dict['break_compliance'],
        shift=input_dict['shift'],
        zone=input_dict['zone'],
        ## model outputs
        anomaly_binary=machines['anomaly_binary'].iloc[0],
        incident_proba=machines['incident_proba'].iloc[0],
        severity=machines['severity'].iloc[0],
        incident_type=machines['incident_type'].iloc[0],
        scores=machines['final_score'].iloc[0],
        ## final decision
        status=machines['status'].iloc[0],
        reason=machines['reason'].iloc[0],
         ## action 
        action_taken=machines['action'].iloc[0],
        timeframe=machines['timeframe'].iloc[0]
    )
    return {
        "anomaly_binary":str(machines['anomaly_flag'].iloc[0]),
        "incident_proba":float(machines['incident_proba'].iloc[0]),
        "severity":str(machines['severity'].iloc[0]),
        "incident_type":str(machines['incident_type'].iloc[0]),
        "scores":float(machines['final_score'].iloc[0]),
        "status":str(machines['status'].iloc[0]),
        "reason":str(machines['reason'].iloc[0]),
        "action":str(machines['action'].iloc[0]),
        "timeframe":str(machines['timeframe'].iloc[0]),
        "shap_explanation":shap_explanation
    }
    

