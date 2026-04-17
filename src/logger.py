import mlflow
from datetime import datetime
import os
import dagshub
from dotenv import load_dotenv

load_dotenv()
os.environ['MLFLOW_TRACKING_PASSWORD']=os.getenv('DAGSHUB_TOKEN')
os.environ['MLFLOW_TRACKING_USERNAME']="justusomari-glitch"
dagshub.init(
    repo_owner="justusomari-glitch",
    repo_name="Health and Safety Management System",
    mlflow=True
)


TRACKING_URI="https://dagshub.com/justusomari-glitch/Industrial_HSE_System.mlflow"
EXPERIMENT_NAME= "Health and Safety Management System"

def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def log_prediction(
    temperature,
    humidity,
    noise_level,
    gas_level,
    vibration,
    voltage,
    pressure,
    co_ppm,
    smoke_level,
    hours_worked,
    days_consecutive,
    ppe_compliance,
    break_compliance,
    shift,
    zone,
    anomaly_binary,
    incident_proba,
    severity,
    incident_type,
    scores,
    rule_engine,
    score_engine
):
    timestamp=datetime.now().strftime("%Y/%m/%d_%H%M%S")
    run_name=f"inference_{temperature}-{timestamp}"
    with mlflow.start_run(run_name="run_name"):
        mlflow.log_metric("temperature",temperature)
        mlflow.log_metric("humidity",humidity)
        mlflow.log_metric("noise_level",noise_level)
        mlflow.log_metric("gas_level",gas_level)
        mlflow.log_metric("vibration",vibration)
        mlflow.log_metric("voltage",voltage)
        mlflow.log_metric("pressure",pressure)
        mlflow.log_metric("co_ppm",co_ppm)
        mlflow.log_metric("hours_worked",hours_worked)
        mlflow.log_metric("days_consecutive",days_consecutive)
        mlflow.log_metric("ppe_compliance",ppe_compliance)
        mlflow.log_metric("break_compliance",break_compliance)
        mlflow.log_metric("smoke_level",smoke_level)
        mlflow.log_param("shift",shift)
        mlflow.log_param("zone",zone)
        mlflow.log_param("severity",severity)
        mlflow.log_param("incident_type",incident_type)
        mlflow.log_param("anomaly_binary",anomaly_binary)
        mlflow.log_metric("incident_proba",incident_proba)
        mlflow.log_metric("scores",scores)
        mlflow.set_param("rule_engine",rule_engine)
        mlflow.set_param('score_engine',score_engine)
        mlflow.set_tag("rule_engine",rule_engine)
        mlflow.set_tag('score_engine',score_engine)

   


