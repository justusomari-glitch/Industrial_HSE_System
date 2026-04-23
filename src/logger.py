import mlflow
from datetime import datetime
import os
import dagshub
from dotenv import load_dotenv


TRACKING_URI="https://dagshub.com/justusomari-glitch/Industrial_HSE_System.mlflow"
EXPERIMENT_NAME= "Health and Safety Management System"


def setup_mlflow():
    import os
    import dagshub
    import mlflow
    from dotenv import load_dotenv
    load_dotenv()
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    try:
        if dagshub_token:
            os.environ['MLFLOW_TRACKING_USERNAME']="justusomari-glitch"
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            dagshub.init(
                repo_owner="justusomari-glitch",
                repo_name="Industrial_HSE_System",
                mlflow=True
            )
            
    except Exception as e:
        print(f"Mlflow setup warning: {e}")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
## we are atarting the mlflow
setup_mlflow()

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
    status,
    reason,
    action_taken,
    timeframe
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
        mlflow.log_param("status",status)
        mlflow.log_param("reason",reason)
        mlflow.log_param("action_taken",action_taken)
        mlflow.log_param("timeframe",timeframe)
        mlflow.set_tag("status",status)
        mlflow.set_tag('reason',reason)

   


