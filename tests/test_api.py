"""API tests. A stub predictor is injected, so no trained artifact or database is needed."""
import numpy as np
from fastapi.testclient import TestClient

from olist_ml.api.main import EXAMPLE_ORDER, app, get_predictor
from olist_ml.pipeline import features_from_orders


class StubPredictor:
    threshold = 0.5
    metadata = {"model": "stub"}

    def predict_proba(self, orders):
        return np.full(len(orders), 0.7)


app.dependency_overrides[get_predictor] = lambda: StubPredictor()
client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_probability_and_decision():
    response = client.post("/predict", json=EXAMPLE_ORDER)
    assert response.status_code == 200
    assert response.json() == {"probability": 0.7, "late": True, "threshold": 0.5}


def test_batch_prediction():
    response = client.post("/predict/batch", json={"orders": [EXAMPLE_ORDER, EXAMPLE_ORDER]})
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_rejects_unknown_state():
    response = client.post("/predict", json={**EXAMPLE_ORDER, "customer_state": "XX"})
    assert response.status_code == 422


def test_rejects_delivery_date_before_purchase():
    response = client.post("/predict", json={**EXAMPLE_ORDER, "estimated_delivery_date": "2018-01-01"})
    assert response.status_code == 422


def test_derived_features_follow_sql_definitions():
    row = features_from_orders([EXAMPLE_ORDER]).iloc[0]
    assert row["purchase_month"] == 6
    assert row["purchase_weekday"] == 5          # 2018-06-15 was a Friday (ISO weekday 5)
    assert row["purchase_hour"] == 14
    assert row["estimated_days"] == 20           # 15 June -> 5 July
    assert not row["same_state"]                 # RJ customer, SP seller