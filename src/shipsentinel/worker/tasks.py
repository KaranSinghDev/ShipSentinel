from shipsentinel.worker.celery_app import celery_app


@celery_app.task(name="shipsentinel.train")
def train_model():
    """Trigger a model training run. Full implementation in Phase 2."""
    raise NotImplementedError("Training task not yet implemented")
