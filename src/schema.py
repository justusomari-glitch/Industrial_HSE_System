from pydantic import BaseModel
import joblib


## loading the models
def load_models():
    anomaly_model=joblib.load("models/anomaly_detection.pkl")
    incident_model=joblib.load("models/incident_model.pkl")
    incident_severity=joblib.load("models/incident_severity_model.pkl")
    incident_type=joblib.load("models/incident_type_model.pkl")
    return anomaly_model,incident_model,incident_severity,incident_type
anomaly_model,incident_model,incident_severity,incident_type=load_models()
print("Models Loaded")
print("Anomaly Model Features:",anomaly_model.feature_names_in_)
print("Incident Model Features:",incident_model.feature_names_in_)
print("Incident Severity Features:",incident_severity.feature_names_in_)
print("Incident Type Features:",incident_type.feature_names_in_)

class AnomalyModel(BaseModel):
    temperature: float
    humidity: float
    noise_level: float
    gas_level: float
    vibration: float
    voltage: float
    pressure: float
    co_ppm: float
    smoke_level: float

class IncidentModel(BaseModel):
    temperature: float
    noise_level: float
    gas_level: float
    vibration: float
    voltage: float
    pressure: float
    co_ppm: float
    smoke_level: float
    hours_worked: float
    days_consecutive: float
    ppe_compliance: float
    break_compliance: float
    shift: str
    zone: str

class IncidentSeverityModel(BaseModel):
    temperature: float
    noise_level: float
    gas_level: float
    vibration: float
    voltage: float
    pressure: float
    co_ppm: float
    smoke_level: float
    hours_worked: float
    days_consecutive: float
    ppe_compliance: float
    break_compliance: float
    shift: str
    zone: str

class IncidentSeverityModel(BaseModel):
    temperature: float
    noise_level: float
    gas_level: float
    vibration: float
    voltage: float
    pressure: float
    co_ppm: float
    smoke_level: float
    hours_worked: float
    days_consecutive: float
    ppe_compliance: float
    break_compliance: float
    shift: str
    zone: str

class HealthAndSafety(BaseModel):
    temperature: float
    humidity: float
    noise_level: float
    gas_level: float
    vibration: float
    voltage: float
    pressure: float
    co_ppm: float
    smoke_level: float
    hours_worked: float
    days_consecutive: float
    ppe_compliance: float
    break_compliance: float
    shift: str
    zone: str
   



