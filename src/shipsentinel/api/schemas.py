from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ShipmentCreate(BaseModel):
    id: str
    carrier: str
    origin: str
    destination: str
    service_type: str
    customer_tier: str
    distance_km: float = Field(gt=0)
    weight_kg: float = Field(gt=0)
    shipment_date: datetime
    scheduled_delivery: datetime


class ShipmentOutcome(BaseModel):
    actual_delivery: datetime
    sla_breached: bool


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    shipment_id: str
    breach_probability: float
    breach_predicted: bool
    model_version: str
    predicted_at: datetime


class TrainingRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    model_version: str
    status: str
    started_at: datetime
    completed_at: datetime | None
    metrics: dict | None
    n_train_samples: int | None
