from fastapi import FastAPI
from shipsentinel.api.routes import shipments, predictions

app = FastAPI(
    title="ShipSentinel",
    description="Logistics SLA Breach Predictor — predict delivery delays before they happen.",
    version="0.1.0",
)

app.include_router(shipments.router)
app.include_router(predictions.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "shipsentinel"}
