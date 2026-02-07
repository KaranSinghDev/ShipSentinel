"""
Prediction and training routes.
POST /predictions/{shipment_id}  — get breach probability for a shipment
POST /training/start             — enqueue a background Celery training job
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from shipsentinel.config import Settings, get_settings
from shipsentinel.db.session import get_db
from shipsentinel.db.models import Shipment, Prediction, TrainingRun
from shipsentinel.api.schemas import PredictionResponse, TrainingRunResponse
from shipsentinel.ml.features import build_feature_row
from shipsentinel.ml.predictor import Predictor, ModelNotLoadedError

router = APIRouter(tags=["predictions"])


@router.post("/predictions/{shipment_id}", response_model=PredictionResponse, status_code=201)
def predict(
    shipment_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Return breach probability for an existing shipment."""
    shipment = db.get(Shipment, shipment_id)
    if not shipment:
        raise HTTPException(status_code=404, detail="Shipment not found")

    predictor = Predictor.get()
    try:
        feature_row = build_feature_row(shipment.__dict__)
        prob, predicted = predictor.predict(feature_row, settings)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    prediction = Prediction(
        shipment_id=shipment_id,
        breach_probability=prob,
        breach_predicted=predicted,
        model_version=predictor.model_version,
        feature_snapshot=feature_row,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


@router.post("/training/start", response_model=TrainingRunResponse, status_code=202)
def trigger_training(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """
    Enqueue a background Celery training job.
    Returns the TrainingRun record immediately (status=running).
    Poll GET /training/{id} or check MLflow for completion.
    """
    import uuid
    from shipsentinel.worker.tasks import train_model

    run = TrainingRun(model_version=f"v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}")
    db.add(run)
    db.commit()
    db.refresh(run)

    train_model.delay(run.id)
    return run
