import numpy as np

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
RISK_LEVELS = {
    "LOW RISK!!": 1,
    "MODERATE RISK!!": 2,
    "HIGH RISK!!": 3,
    "CRITICAL RISK!!": 4
}
## MCDM SCORING
def compute_mcdm_score(anomaly_flag,incident_proba,severity,incident_type):
    anomaly_binary=np.where(anomaly_flag==-1,1,0)
    severity_score=np.array([severity_weights[m] for m in severity])
    type_score=np.array([type_weights[y] for y in incident_type])
    criteria=np.column_stack([anomaly_binary,incident_proba,severity_score,type_score])
    weights=([0.1,0.5,0.3,0.1])
    scores=np.dot(criteria,weights)[0]
    return round(scores,2)
## start of the decision engine
def score_engine(row):
    return row['scores']
#applying soft rules
def apply_soft_rules(scores,row):
    if row['anomaly_binary'] == 1:
        scores +=0.05
    if row['severity'] == "High":
        scores +=0.05
    return min(scores,1.0)
## applying the hard rules
def check_hard_rules(row):
    if row['incident_proba']>0.8 and row['severity'] == "High":
        return "CRITICAL RISK!!","Critical probability and Severity."
    if row['anomaly_binary']== 1 and row['incident_proba']>0.7:
        return "HIGH RISK!!", "Anomaly Detected with High Incident Probability."
    return None,None
## decision engine
def rule_engine(row):
    scores =score_engine(row)
        # apply the soft rules
    scores=apply_soft_rules(scores,row)
    overide_status,reason=check_hard_rules(row)
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
## action mapping based on status
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