"""
Feature engineering for SLA breach prediction.
All functions are pure — no side effects — so they are fully unit-testable.
"""
from datetime import datetime
import pandas as pd

CATEGORICAL_FEATURES = ["carrier", "origin", "destination", "service_type", "customer_tier"]
NUMERIC_FEATURES = [
    "distance_km", "weight_kg", "sla_window_hours",
    "shipment_hour", "shipment_dow", "shipment_month",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def compute_sla_window_hours(shipment_date: datetime, scheduled_delivery: datetime) -> float:
    """Hours between shipment creation and promised delivery date."""
    return (scheduled_delivery - shipment_date).total_seconds() / 3600


def extract_temporal_features(dt: datetime) -> dict:
    """Hour of day, day of week (0=Mon), month — all carry delay signal."""
    return {
        "shipment_hour": dt.hour,
        "shipment_dow": dt.weekday(),
        "shipment_month": dt.month,
    }


def build_feature_row(shipment: dict) -> dict:
    """
    Convert a raw shipment dict to a flat feature dict ready for LightGBM.
    Expected keys: carrier, origin, destination, service_type, customer_tier,
                   distance_km, weight_kg, shipment_date, scheduled_delivery.
    """
    return {
        "carrier": shipment["carrier"],
        "origin": shipment["origin"],
        "destination": shipment["destination"],
        "service_type": shipment["service_type"],
        "customer_tier": shipment["customer_tier"],
        "distance_km": shipment["distance_km"],
        "weight_kg": shipment["weight_kg"],
        "sla_window_hours": compute_sla_window_hours(
            shipment["shipment_date"], shipment["scheduled_delivery"]
        ),
        **extract_temporal_features(shipment["shipment_date"]),
    }


def build_feature_dataframe(shipments: list[dict]) -> pd.DataFrame:
    """Build a feature matrix from a list of shipment dicts."""
    df = pd.DataFrame([build_feature_row(s) for s in shipments])
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df[ALL_FEATURES]
