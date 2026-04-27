import joblib
models_loaded=False
anomaly_model=None
incident_model=None
incident_severity=None
incident_type_model=None

def load_models():
    global anomaly_model,incident_model,incident_severity,incident_type_model,models_loaded
    if models_loaded:
        return
    anomaly_model=joblib.load("models/anomaly_detection.pkl")
    incident_model=joblib.load("models/incident_model.pkl")
    incident_severity=joblib.load("models/incident_severity_model.pkl")
    incident_type_model=joblib.load("models/incident_type_model.pkl")
    models_loaded = True