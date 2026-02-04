"""
Data access for ML training: queries labelled shipments from PostgreSQL.
Pure DB layer — no ML logic here.
"""
from sqlalchemy.orm import Session
from shipsentinel.db.models import Shipment

MIN_TRAINING_SAMPLES = 50


class InsufficientDataError(Exception):
    """Raised when there are not enough labelled shipments to train a model."""


def get_labelled_shipments(session: Session) -> list[dict]:
    """
    Return all shipments where sla_breached IS NOT NULL (i.e. outcome recorded).
    Each dict contains all feature fields + the label.

    Raises InsufficientDataError when < MIN_TRAINING_SAMPLES exist.
    Record outcomes via PATCH /shipments/{id}/outcome.
    """
    rows = (
        session.query(Shipment)
        .filter(Shipment.sla_breached.isnot(None))
        .all()
    )
    if len(rows) < MIN_TRAINING_SAMPLES:
        raise InsufficientDataError(
            f"Need at least {MIN_TRAINING_SAMPLES} labelled shipments to train, "
            f"got {len(rows)}. Record outcomes via PATCH /shipments/{{id}}/outcome."
        )
    return [
        {
            "carrier": s.carrier,
            "origin": s.origin,
            "destination": s.destination,
            "service_type": s.service_type,
            "customer_tier": s.customer_tier,
            "distance_km": s.distance_km,
            "weight_kg": s.weight_kg,
            "shipment_date": s.shipment_date,
            "scheduled_delivery": s.scheduled_delivery,
            "sla_breached": bool(s.sla_breached),
        }
        for s in rows
    ]
