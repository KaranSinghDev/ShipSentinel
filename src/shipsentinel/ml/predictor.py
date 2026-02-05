"""
Singleton breach predictor — loads the LightGBM model from MLflow registry.
Thread-safe via double-checked locking.
Call Predictor.reset() after training so the next request reloads the new model.
"""
import threading

import mlflow.sklearn
import pandas as pd

from shipsentinel.config import Settings
from shipsentinel.ml.features import ALL_FEATURES, CATEGORICAL_FEATURES


class ModelNotLoadedError(Exception):
    """Raised when predict() cannot load a model (registry empty)."""


class Predictor:
    _instance: "Predictor | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None          # sklearn-compatible LGBMClassifier
        self._model_version: str | None = None

    # --- Singleton lifecycle ---

    @classmethod
    def get(cls) -> "Predictor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Invalidate singleton so the next predict() reloads from registry."""
        with cls._lock:
            cls._instance = None

    # --- Model loading ---

    def load(self, settings: Settings) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        model_uri = f"models:/{settings.model_registry_name}/latest"
        self._model = mlflow.sklearn.load_model(model_uri)
        self._model_version = settings.model_registry_name

    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_version(self) -> str:
        return self._model_version or "unknown"

    # --- Inference ---

    def predict(self, feature_row: dict, settings: Settings) -> tuple[float, bool]:
        """
        Returns (breach_probability, breach_predicted).
        Lazily loads the model from MLflow on the first call.

        Raises ModelNotLoadedError if no registered model exists.
        """
        if not self.is_loaded():
            try:
                self.load(settings)
            except Exception as exc:
                raise ModelNotLoadedError(
                    f"Could not load model '{settings.model_registry_name}' "
                    f"from MLflow ({settings.mlflow_tracking_uri}): {exc}"
                ) from exc

        df = pd.DataFrame([feature_row])
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype("category")
        df = df[ALL_FEATURES]

        prob = float(self._model.predict_proba(df)[0, 1])
        predicted = prob >= settings.sla_breach_threshold
        return prob, predicted
