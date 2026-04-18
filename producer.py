from confluent_kafka import Producer
import json
import time
import random
from dotenv import load_dotenv
import os

load_dotenv()



conf={
    'bootstrap.servers':os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
    'sasl.mechanism':'PLAIN',
    'security.protocol':'SASL_SSL',
    'sasl.username':os.getenv('KAFKA_API_KEY'),
    'sasl.password':os.getenv('KAFKA_API_SECRET')
}
producer=Producer(conf)
## To insert a debug message
def delivery_report(err,msg):
    if err:
        print(err)
    else:
        print (f"{msg.topic()}[{msg.partition()}]")



def generate_random_data():
    data={
        'temperature':round(random.uniform(10.0,150.0),2),
        'humidity':round(random.uniform(10.0,150.0),2),
        'noise_level':round(random.uniform(20.0,170.0),2),
        'gas_level':round(random.uniform(10.0,1000.0),2), 
        'vibration':round(random.uniform(0.0,40.0),2),
        'voltage':round(random.uniform(110.0,400.0),2),
        'pressure':round(random.uniform(50.0,300.0),2),
        'co_ppm':round(random.uniform(0.0,300.0),2),
        'smoke_level':round(random.uniform(10.0,500.0),2),
        'hours_worked':round(random.uniform(2.0,15.0),2),
        'days_consecutive':random.uniform(1,20),
        'ppe_compliance':round(random.uniform(0.0,1.0),2),
        'break_compliance':random.choice([0,1]),
        'shift':random.choice(['Morning','Afternoon','Night']),
        'zone':random.choice(['Zone_B_Moderate', 'Zone_D_HighRisk', 'Zone_C_Dangerous','Zone_A_Safe'])
        }
    return data
    
print("Producer Started")
while True:
    data=generate_random_data()
    producer.produce('machine_sesnsor_data',
                     value=json.dumps(data).encode('utf-8'),
                     callback=delivery_report
                    )
    producer.poll(7)
    print(f'sent:{data}')
    time.sleep(3)
producer.flush()