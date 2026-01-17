from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from shipsentinel.db.session import get_db
from shipsentinel.db.models import Shipment
from shipsentinel.api.schemas import ShipmentCreate, ShipmentOutcome

router = APIRouter(prefix="/shipments", tags=["shipments"])


@router.post("/", status_code=201)
def create_shipment(payload: ShipmentCreate, db: Session = Depends(get_db)):
    if db.get(Shipment, payload.id):
        raise HTTPException(status_code=409, detail="Shipment already exists")
    shipment = Shipment(**payload.model_dump())
    db.add(shipment)
    db.commit()
    db.refresh(shipment)
    return shipment


@router.patch("/{shipment_id}/outcome")
def record_outcome(shipment_id: str, outcome: ShipmentOutcome, db: Session = Depends(get_db)):
    shipment = db.get(Shipment, shipment_id)
    if not shipment:
        raise HTTPException(status_code=404, detail="Shipment not found")
    shipment.actual_delivery = outcome.actual_delivery
    shipment.sla_breached = outcome.sla_breached
    db.commit()
    db.refresh(shipment)
    return shipment


@router.get("/{shipment_id}")
def get_shipment(shipment_id: str, db: Session = Depends(get_db)):
    shipment = db.get(Shipment, shipment_id)
    if not shipment:
        raise HTTPException(status_code=404, detail="Shipment not found")
    return shipment
