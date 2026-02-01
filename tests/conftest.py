import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from shipsentinel.main import app
from shipsentinel.db.session import Base, get_db


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(eng)
    yield eng


@pytest.fixture
def db(engine):
    """
    Each test gets a fresh transaction that is rolled back on teardown.
    This guarantees isolation without recreating the schema.
    """
    connection = engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection, autocommit=False, autoflush=False)
    session = Session()
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db):
    def override_get_db():
        yield db
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_shipment_payload():
    now = datetime.utcnow()
    return {
        "id": "SHIP-001",
        "carrier": "FedEx",
        "origin": "Mumbai",
        "destination": "Delhi",
        "service_type": "express",
        "customer_tier": "gold",
        "distance_km": 1400.0,
        "weight_kg": 5.5,
        "shipment_date": now.isoformat(),
        "scheduled_delivery": (now + timedelta(days=2)).isoformat(),
    }
