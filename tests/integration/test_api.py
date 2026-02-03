"""
Integration tests for ShipSentinel REST API.
Uses SQLite via conftest fixtures. Defines the API contract.
"""
from datetime import datetime, timedelta
import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_shipment(client, sample_shipment_payload):
    r = client.post("/shipments/", json=sample_shipment_payload)
    assert r.status_code == 201
    body = r.json()
    assert body["id"] == "SHIP-001"
    assert body["carrier"] == "FedEx"
    assert body["sla_breached"] is None


def test_create_shipment_duplicate_409(client, sample_shipment_payload):
    client.post("/shipments/", json=sample_shipment_payload)
    r = client.post("/shipments/", json=sample_shipment_payload)
    assert r.status_code == 409


def test_get_shipment(client, sample_shipment_payload):
    client.post("/shipments/", json=sample_shipment_payload)
    r = client.get("/shipments/SHIP-001")
    assert r.status_code == 200
    assert r.json()["id"] == "SHIP-001"


def test_get_shipment_not_found(client):
    r = client.get("/shipments/NONEXISTENT")
    assert r.status_code == 404


def test_record_outcome(client, sample_shipment_payload):
    client.post("/shipments/", json=sample_shipment_payload)
    actual = (datetime.utcnow() + timedelta(days=3)).isoformat()
    r = client.patch("/shipments/SHIP-001/outcome", json={
        "actual_delivery": actual,
        "sla_breached": True,
    })
    assert r.status_code == 200
    assert r.json()["sla_breached"] is True


def test_record_outcome_not_found(client):
    r = client.patch("/shipments/GHOST/outcome", json={
        "actual_delivery": datetime.utcnow().isoformat(),
        "sla_breached": False,
    })
    assert r.status_code == 404


def test_predict_no_model_503(client, sample_shipment_payload):
    client.post("/shipments/", json=sample_shipment_payload)
    r = client.post("/predictions/SHIP-001")
    assert r.status_code == 503


def test_predict_nonexistent_shipment_404(client):
    r = client.post("/predictions/GHOST-999")
    assert r.status_code == 404


def test_shipment_missing_required_field(client):
    payload = {"id": "SHIP-BAD", "carrier": "FedEx"}
    r = client.post("/shipments/", json=payload)
    assert r.status_code == 422
