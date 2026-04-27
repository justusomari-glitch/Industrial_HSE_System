from pydantic import BaseModel

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
    days_consecutive: int
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
    days_consecutive: int
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
    days_consecutive: int
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
    days_consecutive: int
    ppe_compliance: float
    break_compliance: float
    shift: str
    zone: str
   



