from datetime import datetime
from sqlalchemy import String, Float, Boolean, DateTime, Integer, Text, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from shipsentinel.db.session import Base


class Shipment(Base):
    __tablename__ = "shipments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    carrier: Mapped[str] = mapped_column(String(64))
    origin: Mapped[str] = mapped_column(String(128))
    destination: Mapped[str] = mapped_column(String(128))
    service_type: Mapped[str] = mapped_column(String(64))   # express | standard | economy
    customer_tier: Mapped[str] = mapped_column(String(32))  # gold | silver | bronze
    distance_km: Mapped[float] = mapped_column(Float)
    weight_kg: Mapped[float] = mapped_column(Float)
    shipment_date: Mapped[datetime] = mapped_column(DateTime)
    scheduled_delivery: Mapped[datetime] = mapped_column(DateTime)
    actual_delivery: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    sla_breached: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="shipment")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    shipment_id: Mapped[str] = mapped_column(String(64), ForeignKey("shipments.id"))
    breach_probability: Mapped[float] = mapped_column(Float)
    breach_predicted: Mapped[bool] = mapped_column(Boolean)
    model_version: Mapped[str] = mapped_column(String(64))
    predicted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    feature_snapshot: Mapped[dict] = mapped_column(JSON)

    shipment: Mapped["Shipment"] = relationship(back_populates="predictions")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(64), unique=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running")  # running | completed | failed
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    n_train_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __init__(self, **kwargs):
        kwargs.setdefault("status", "running")
        super().__init__(**kwargs)
