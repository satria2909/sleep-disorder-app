from pydantic import BaseModel

class PredictionRequest(BaseModel):

    Age:int
    Sleep_Duration:float
    Quality_of_Sleep:int
    Physical_Activity_Level:int
    Stress_Level:int
    Heart_Rate:int
    Daily_Steps:int
    Systolic_BP:int
    Diastolic_BP:int