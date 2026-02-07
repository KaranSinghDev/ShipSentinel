"""
Celery tasks for ShipSentinel.
The train_model task runs asynchronously so training doesn't block the API.
"""
from datetime import datetime

from shipsentinel.worker.celery_app import celery_app


@celery_app.task(name="shipsentinel.train", bind=True)
def train_model(self, run_id: int) -> dict:
    """
    Background training task.
    1. Open a DB session
    2. Call ml/trainer.train() — fetches data, trains LightGBM, logs to MLflow
    3. Update the TrainingRun record with status + metrics
    4. Invalidate the Predictor singleton so the next request reloads the new model
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from shipsentinel.config import Settings
    from shipsentinel.db.models import TrainingRun
    from shipsentinel.ml.trainer import train
    from shipsentinel.ml.predictor import Predictor

    settings = Settings()
    engine = create_engine(settings.database_url)

    with Session(engine) as session:
        run = session.get(TrainingRun, run_id)
        try:
            result = train(session, settings)
            run.status = "completed"
            run.metrics = {"auc": result["auc"]}
            run.n_train_samples = result["n_samples"]
            run.completed_at = datetime.utcnow()
            session.commit()
            Predictor.reset()
            return result
        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)
            run.completed_at = datetime.utcnow()
            session.commit()
            raise
