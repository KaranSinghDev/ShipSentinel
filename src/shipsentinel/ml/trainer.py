"""
LightGBM trainer with MLflow tracking.
Trains a binary SLA breach probability model on historical shipment data.
Uses 5-fold stratified cross-validation to compute an honest AUC estimate
before fitting the final model on all data.
"""
import uuid
from datetime import datetime

import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sqlalchemy.orm import Session

from shipsentinel.config import Settings
from shipsentinel.ml.data import get_labelled_shipments
from shipsentinel.ml.features import build_feature_dataframe, CATEGORICAL_FEATURES

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "verbose": -1,
    "random_state": 42,
}


def _make_version() -> str:
    return f"v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"


def train(session: Session, settings: Settings) -> dict:
    """
    Train LightGBM on all labelled shipments.

    Returns:
        {
            "auc": float,           mean OOF AUC across 5 folds
            "mlflow_run_id": str,
            "n_samples": int,
            "model_version": str,   timestamp + uuid slug
        }
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    shipments = get_labelled_shipments(session)
    df = build_feature_dataframe(shipments)
    y = np.array([s["sla_breached"] for s in shipments], dtype=int)

    model_version = _make_version()

    with mlflow.start_run(run_name=model_version) as run:
        mlflow.log_params({**LGBM_PARAMS, "n_samples": len(shipments)})

        # 5-fold OOF AUC — gives honest performance estimate without a held-out test set
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y))

        for train_idx, val_idx in skf.split(df, y):
            fold_clf = lgb.LGBMClassifier(**LGBM_PARAMS)
            fold_clf.fit(
                df.iloc[train_idx], y[train_idx],
                categorical_feature=CATEGORICAL_FEATURES,
            )
            oof_probs[val_idx] = fold_clf.predict_proba(df.iloc[val_idx])[:, 1]

        mean_auc = float(roc_auc_score(y, oof_probs))
        mlflow.log_metric("cv_auc", mean_auc)

        # Final model trained on all labelled data
        final_clf = lgb.LGBMClassifier(**LGBM_PARAMS)
        final_clf.fit(df, y, categorical_feature=CATEGORICAL_FEATURES)

        mlflow.sklearn.log_model(
            final_clf,
            artifact_path="model",
            registered_model_name=settings.model_registry_name,
        )
        mlflow.set_tag("model_version", model_version)

    return {
        "auc": mean_auc,
        "mlflow_run_id": run.info.run_id,
        "n_samples": len(shipments),
        "model_version": model_version,
    }
