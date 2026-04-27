import os
from src.schema import HealthAndSafety
from fastapi import FastAPI
import pandas as pd
import numpy as np
from src.logger import log_prediction,setup_mlflow
import dagshub
from src.models import load_models,anomaly_model,incident_model,incident_severity,incident_type_model
from src.mcdm_scoring import compute_mcdm_score,rule_engine,action_mapping,check_hard_rules,apply_soft_rules,score_engine,RISK_LEVELS
from src.explainability import get_shap_explanation,get_shap_sensor_explanation
from src.llm import get_llm_explanations
from src import models as model_store


app=FastAPI()
@app.on_event('startup')
def startup_event():
    setup_mlflow()



@app.get("/")
def home():
    return {"Message":"Health and Safety System Is up and running"}

@app.post("/predict")
def predict(data:HealthAndSafety):
    load_models()
## reload the globals after loading the models
    anomaly_model=model_store.anomaly_model
    incident_model=model_store.incident_model
    incident_severity=model_store.incident_severity
    incident_type_model=model_store.incident_type_model


    input_dict=data.model_dump()
    df=pd.DataFrame([input_dict])

    ## make predictions
    anomaly_flag=anomaly_model.predict(df)
    incident_proba=incident_model.predict_proba(df)[:,1]
    severity=incident_severity.predict(df)
    incident_type=incident_type_model.predict(df)

    ## compute MCDM score
    anomaly_binary=np.where(anomaly_flag==-1,1,0)
    scores=compute_mcdm_score(anomaly_flag,incident_proba,severity,incident_type)


    ## shap explainability for MCDM
    weights=([0.1,0.5,0.3,0.1])
    severity_weights={
    "Medium":0.8,
    "High":1.0,
    "Low":0.4,
    "None":0.1
    }
    type_weights={
    "Chemical":0.8,
    "Electrical":0.75,
    "Fire":1.0,
    "Mechanical":0.6,
    "None":0.1
    }
    severity_score=np.array([severity_weights[m] for m in severity])
    type_score=np.array([type_weights[y] for y in incident_type])
    criteria=np.column_stack([anomaly_binary,incident_proba,severity_score,type_score])
    shap_explanation=get_shap_explanation(criteria,weights)
    shap_sensor_explanation=get_shap_sensor_explanation(anomaly_model,df)

    # build machines dataframe
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

    llm_explanation=get_llm_explanations(input_dict,machines)

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
        "shap_explanation":shap_explanation,
        "shap_sensor_explanation":shap_sensor_explanation,
        "llm_explanation": llm_explanation
    }
    

