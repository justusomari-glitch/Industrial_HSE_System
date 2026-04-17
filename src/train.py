from logger import setup_mlflow
import mlflow
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np



models_loaded = False
def load_models():
    global anomaly_model,incident_model,incident_severity,incident_type,models_loaded
    if models_loaded:
        return
    anomaly_model=joblib.load("models/anomaly_detection.pkl")
    incident_model=joblib.load("models/incident_model.pkl")
    incident_severity=joblib.load("models/incident_severity_model.pkl")
    incident_type=joblib.load("models/incident_type_model.pkl")
    models_loaded = True

df=pd.read_csv(r"C:\Users\user\Desktop\Industrial_HSE_System\hse dataset 300k.csv")
df=df.dropna()
anomaly=df['anomaly']
def run_mlflow_logging(
        anomaly,
        anomaly_preds,
        y_test_incident,
        incident_preds,
        y_test_severity,
        severity_preds,
        y_test_type,
        type_preds,
        incident_proba,
        incident_type,
        anomaly_model,
        incident_model,
        incident_severity,
):
    setup_mlflow()
    with mlflow.start_run(run_name="anomaly_model"):
        mlflow.log_param('model_type',"IsolationForest")
        mlflow.log_metric("f1_score",f1_score(df['anomaly'],anomaly_preds,zero_division=0))
        mlflow.log_metric("accuracy_score",accuracy_score(df['anomaly'],anomaly_preds))
        mlflow.log_metric("recall_score",recall_score(df['anomaly'],anomaly_preds,zero_division=0))
        mlflow.log_metric("precision_score",precision_score(df['anomaly'],anomaly_preds,zero_division=0))
        mlflow.sklearn.log_model(anomaly_model,artifact_path='anomaly_model')
    with mlflow.start_run(run_name="incident_model"):
        mlflow.log_param('model_type',"XGBClassidier")
        mlflow.log_metric("f1_score",f1_score(y_test_incident,incident_preds))
        mlflow.log_metric("accuracy_score",accuracy_score(y_test_incident,incident_preds))
        mlflow.log_metric("recall_score",recall_score(y_test_incident,incident_preds))
        mlflow.log_metric("precision_score",precision_score(y_test_incident,incident_preds))
        mlflow.sklearn.log_model(incident_model,artifact_path='incident_model')
    with mlflow.start_run(run_name="severity_model"):
        mlflow.log_param('model_type',"Random Forest Classifier")
        mlflow.log_metric("f1_score",f1_score(y_test_severity,severity_preds,average='weighted'))
        mlflow.log_metric("accuracy_score",accuracy_score(y_test_severity,severity_preds))
        mlflow.log_metric("recall_score",recall_score(y_test_severity,severity_preds,average='weighted'))
        mlflow.log_metric("precision_score",precision_score(y_test_severity,severity_preds,average='weighted'))
        mlflow.sklearn.log_model(incident_severity,artifact_path='incident_severity')
    with mlflow.start_run(run_name="incident_type_model"):
        mlflow.log_param('model_type',"Random Forest Classifier")
        mlflow.log_metric("f1_score",f1_score(y_test_type,type_preds,zero_division=0,average='weighted'))
        mlflow.log_metric("accuracy_score",accuracy_score(y_test_type,type_preds))
        mlflow.log_metric("recall_score",recall_score(y_test_type,type_preds,zero_division=0,average='weighted'))
        mlflow.log_metric("precision_score",precision_score(y_test_type,type_preds,zero_division=0,average='weighted'))
        mlflow.sklearn.log_model(incident_type,artifact_path='incident_type')

if __name__=="__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import numpy as np

    ## import the table
    df=pd.read_csv(r"C:\Users\user\Desktop\Industrial_HSE_System\hse dataset 300k.csv")
    df=df.dropna()

## defining the input or the independent features
    x_anomaly=df[['temperature','humidity','noise_level','gas_level','vibration','voltage','pressure','co_ppm','smoke_level']]
    x_incident=df[['temperature','noise_level','gas_level','vibration','voltage','pressure','co_ppm','smoke_level','hours_worked','days_consecutive','ppe_compliance','break_compliance','shift','zone']]
    x_severity=df[['temperature','noise_level','gas_level','vibration','voltage','pressure','co_ppm','smoke_level','hours_worked','days_consecutive','ppe_compliance','break_compliance','shift','zone']]
    x_type=df[['temperature','noise_level','gas_level','vibration','voltage','pressure','co_ppm','smoke_level','hours_worked','days_consecutive','ppe_compliance','break_compliance','shift','zone']]
## we now define the dependent features
    y_severity=df['severity']
    y_type=df['incident_type']
    y_incident=df['incident']

## splittng the data acoording to each model

    x_train_anomaly,x_test_anomaly=train_test_split(x_anomaly,test_size=0.3,random_state=42)
    x_train_incident,x_test_incident,y_train_incident,y_test_incident=train_test_split(x_incident,y_incident,test_size=0.3,random_state=42)

    x_train_type,x_test_type,y_train_type,y_test_type=train_test_split(x_type,y_type,test_size=0.3,random_state=42)

    x_train_severity,x_test_severity,y_train_severity,y_test_severity=train_test_split(x_severity,y_severity,test_size=0.3,random_state=42)

    # load the models
    anomaly_model=joblib.load("models/anomaly_detection.pkl")
    incident_model=joblib.load("models/incident_model.pkl")
    incident_severity=joblib.load("models/incident_severity_model.pkl")
    incident_type=joblib.load("models/incident_type_model.pkl")

    #make predictions
    
    anomaly_preds=anomaly_model.predict(x_anomaly)
    anomaly_preds=np.where(anomaly_preds==-1,1,0)
    incident_preds=incident_model.predict(x_test_incident)
    severity_preds=incident_severity.predict(x_test_severity)
    type_preds=incident_type.predict(x_test_incident)
    incident_proba=incident_model.predict_proba(x_test_incident)
# run the loging
    run_mlflow_logging(
        anomaly,
        anomaly_preds,
        y_test_incident,
        incident_preds,
        y_test_severity,
        severity_preds,
        y_test_type,
        type_preds,
        incident_proba,
        incident_type,
        anomaly_model,
        incident_model,
        incident_severity
    )
    
        
