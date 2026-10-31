"""REST API for late-delivery prediction.

Run:  uvicorn olist_ml.api.main:app --reload     Interactive docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from olist_ml.pipeline import LatePredictor

State = Literal["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA",
                "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"]
PaymentType = Literal["credit_card", "boleto", "voucher", "debit_card", "not_defined"]

EXAMPLE_ORDER = {
    "purchased_at": "2018-06-15T14:30:00", "estimated_delivery_date": "2018-07-05",
    "n_items": 1, "n_sellers": 1, "total_price": 129.9, "total_freight": 18.5,
    "total_weight_g": 800, "total_volume_cm3": 6000, "max_photos": 2,
    "main_category": "bed_bath_table", "payment_type": "credit_card", "installments": 3,
    "customer_state": "RJ", "seller_state": "SP", "distance_km": 360,
    "seller_prior_deliveries": 40, "seller_prior_late": 3,
}


class Order(BaseModel):
    """Information available when the order is placed."""

    purchased_at: datetime
    estimated_delivery_date: date
    n_items: int = Field(ge=1, le=100)
    n_sellers: int = Field(ge=1, le=20)
    total_price: float = Field(gt=0)
    total_freight: float = Field(ge=0)
    total_weight_g: float | None = Field(default=None, ge=0)
    total_volume_cm3: float | None = Field(default=None, ge=0)
    max_photos: int | None = Field(default=None, ge=0)
    main_category: str = "unknown"
    payment_type: PaymentType = "credit_card"
    installments: int = Field(default=1, ge=0, le=30)
    customer_state: State
    seller_state: State
    distance_km: float = Field(ge=0, le=5000)
    seller_prior_deliveries: int = Field(default=0, ge=0)
    seller_prior_late: int = Field(default=0, ge=0)

    model_config = {"json_schema_extra": {"examples": [EXAMPLE_ORDER]}}

    @model_validator(mode="after")
    def check_consistency(self) -> Order:
        if self.estimated_delivery_date <= self.purchased_at.date():
            raise ValueError("estimated_delivery_date must be after the purchase date")
        if self.seller_prior_late > self.seller_prior_deliveries:
            raise ValueError("seller_prior_late cannot exceed seller_prior_deliveries")
        if self.n_sellers > self.n_items:
            raise ValueError("n_sellers cannot exceed n_items")
        return self


class BatchRequest(BaseModel):
    orders: list[Order] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    probability: float
    late: bool
    threshold: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", "artifacts/late_predictor.pkl")
    app.state.predictor = LatePredictor.load(path) if os.path.exists(path) else None
    yield


app = FastAPI(title="Olist late-delivery API", version="1.0.0", lifespan=lifespan,
              description="Predicts at purchase time whether an order will arrive after its estimated date.")


def get_predictor(request: Request) -> LatePredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run scripts/train_final.py first.")
    return predictor


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_loaded": getattr(request.app.state, "predictor", None) is not None}


@app.get("/model")
def model_info(predictor: LatePredictor = Depends(get_predictor)) -> dict:
    return predictor.metadata


@app.post("/predict", response_model=Prediction)
def predict(order: Order, predictor: LatePredictor = Depends(get_predictor)) -> Prediction:
    probability = float(predictor.predict_proba([order.model_dump()])[0])
    return Prediction(probability=probability, late=probability >= predictor.threshold,
                      threshold=predictor.threshold)


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: BatchRequest, predictor: LatePredictor = Depends(get_predictor)) -> list[Prediction]:
    probabilities = predictor.predict_proba([order.model_dump() for order in batch.orders])
    return [Prediction(probability=float(p), late=float(p) >= predictor.threshold,
                       threshold=predictor.threshold) for p in probabilities]