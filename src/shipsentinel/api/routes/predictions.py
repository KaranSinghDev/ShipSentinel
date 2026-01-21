from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from shipsentinel.db.session import get_db
from shipsentinel.db.models import Shipment
from shipsentinel.api.schemas import PredictionResponse, TrainingRunResponse

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/{shipment_id}", response_model=PredictionResponse)
def predict(shipment_id: str, db: Session = Depends(get_db)):
    shipment = db.get(Shipment, shipment_id)
    if not shipment:
        raise HTTPException(status_code=404, detail="Shipment not found")
    # Predictor wired in Phase 2
    raise HTTPException(status_code=503, detail="No trained model available")


@router.post("/training/start", response_model=TrainingRunResponse)
def trigger_training(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # Training task wired in Phase 2
    raise HTTPException(status_code=503, detail="Training not yet implemented")
