from pydantic import BaseModel
from typing import Optional

class ClimateDTO(BaseModel):
    Description_Warm: bool
    Rain: int 
    Visibility_km: float
    Wind_Speed_kmh: int
    Wind_Bearing_degrees: float
    Description_Cold: bool
    Humidity: float
    Description_Normal: bool