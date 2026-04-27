import shap
import numpy as np

def get_shap_explanation(criteria,weights):
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
    return shap_explanation

def get_shap_sensor_explanation(anomaly_model,df):
    input_feature_names=["temperature","humidity","noise_level","gas_level","vibration",
                          "voltage","pressure","co_ppm","smoke_level","hours_worked",
                         "days_consecutive","ppe_compliance","break_compliance"]
    input_array=df[input_feature_names].values
    explainer_input=shap.TreeExplainer(anomaly_model.named_steps['iso'])
    shap_values_input=explainer_input.shap_values(input_array)
    if isinstance(shap_values_input, list):
        shap_values_input=shap_values_input[0]
    else:
        shap_values_input=shap_values_input
    shap_sensor_explanation={
        input_feature_names[i]: round(float(shap_values_input[0][i]), 2) 
        for i in range(len(input_feature_names))
    }
    return shap_sensor_explanation