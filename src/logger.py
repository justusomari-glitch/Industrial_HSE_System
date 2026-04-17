import mlflow
from datetime import datetime
import os
import dagshub
from dotenv import load_dotenv


TRACKING_URI="sqlite:///mlflow.db"
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
        mlflow.set_tag("rule_engine",rule_engine)
        mlflow.set_tag('score_engine',score_engine)

   


