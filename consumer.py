from confluent_kafka import Consumer
from src.predict import predict,HealthAndSafety
import os
import json
import pandas
from dotenv import load_dotenv
import pymysql


load_dotenv()

conf={
    'bootstrap.servers':os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
    'sasl.mechanism':'PLAIN',
    'security.protocol':'SASL_SSL',
    'sasl.username':os.getenv('KAFKA_API_KEY'),
    'sasl.password':os.getenv('KAFKA_API_SECRET'),
    'group.id':'hse-consumer-group',
    'auto.offset.reset':'latest'
}

consumer=Consumer(conf)
consumer.subscribe(['machine_sesnsor_data'])
print('Consumer listening')

## we now want to set up the mysql database and host it in aiven

db=pymysql.connect(
    host=os.getenv('DB_HOST'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    database=os.getenv('DB_NAME'),
    port=int(os.getenv('DB_PORT')),
    ssl={"ssl":{'mode':os.getenv("SSL_MODE")}}
)

cursor=db.cursor()

## creating a table in mysql
cursor.execute("""           
CREATE TABLE IF NOT EXISTS healthandsafety(
    id INT AUTO_INCREMENT PRIMARY KEY,
    temperature FLOAT,
    humidity FLOAT,
    noise_level FLOAT,
    gas_level FLOAT,
    vibration FLOAT,
    voltage FLOAT,
    pressure FLOAT,
    co_ppm FLOAT,
    smoke_level FLOAT,
    hours_worked FLOAT,
    days_consecutive INT,
    ppe_compliance FLOAT,
    break_compliance BOOLEAN,
    shift VARCHAR(100),
    zone VARCHAR(100),
    anomaly_binary BOOLEAN,
    incident_proba FLOAT,
    severity VARCHAR(100),
    incident_type VARCHAR(100),
    scores FLOAT,
    rule_engine VARCHAR(250),
    score_engine VARCHAR(250)
)""")
db.commit()
try:
    while True:
        msg=consumer.poll(7)
        if msg is None:
            continue
        if msg.error():
            print("Consumer Error:",msg.error())
            continue
        data=json.loads(msg.value().decode('utf-8'))
        print("Recieved:",data)
        if not data:
            continue
        ## we run prediction and save the data in my sql
        input_data=HealthAndSafety(**data)
        result=predict(input_data)
        if isinstance(result,list):
            result=result[0]
        print ("Prediction Result",result)
        record={
            "temperature": data.get("temperature"), 
            "humidity" : data.get("humidity"),
            "noise_level" : data.get("noise_level"),
            "gas_level" : data.get("gas_level"),
            "vibration" : data.get("vibration"),
            "voltage" : data.get("voltage"),
            "pressure" : data.get("pressure"),
            "co_ppm" : data.get("co_ppm"),
            "smoke_level" : data.get("smoke_level"),
            "hours_worked" : data.get("hours_worked"),
            "days_consecutive" : data.get("days_consecutive"),
            "ppe_compliance" : data.get("ppe_compliance"),
            "break_compliance" : data.get("break_compliance"),
            "shift": data.get("shift"),
            "zone": data.get("zone"),
            "anomaly_binary": result.get("anomaly_binary"),
            "incident_proba": result.get("incident_proba"),
            "severity": result.get("severity"),
            "incident_type": result.get("incident_type"),
            "scores": result.get("scores"),
            "rule_engine": result.get("rule_engine"),
            "score_engine": result.get("score_engine"),

        }
        anomaly_value= 1 if record['anomaly_binary']== 'ANOMALY DETECTED' else 0
        sql= """
        INSERT INTO healthandsafety (
            temperature,humidity,noise_level ,gas_level,vibration,
            voltage,pressure,co_ppm,smoke_level,hours_worked,days_consecutive,ppe_compliance,
            break_compliance,shift,zone,anomaly_binary,incident_proba,severity,
            incident_type,scores,rule_engine,score_engine
        )  VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        values= (
            record["temperature"], 
            record["humidity"],
            record["noise_level"] ,
            record["gas_level"],
            record["vibration"],
            record["voltage"],
            record["pressure"],
            record["co_ppm"],
            record["smoke_level"],
            record["hours_worked"],
            record["days_consecutive"],
            record["ppe_compliance"],
            record["break_compliance"],
            record["shift"],
            record["zone"],
            anomaly_value,
            record["incident_proba"],
            record["severity"],
            record["incident_type"],
            record["scores"],
            record["rule_engine"],
            record["score_engine"]
        )
        print("Placeholders:",sql.count("%s"))
        print("Values:",len(values))
        cursor.execute(sql,values)
        db.commit()
        print("Record Saved To Data Base")
except KeyboardInterrupt:
    print("Consumer Stopped")


        
