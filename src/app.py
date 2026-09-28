from contextlib import asynccontextmanager
import json
from pathlib import Path
import pickle
from typing import List, Optional
import sys

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from src.wrangler import data_wrangler

sys.modules['wrangler'] = sys.modules['src.wrangler']

# Global model container
MODEL_PIPELINE = None
METRICS_DATA = {}

# --- 1. Lifespan Manager (FIXED: Added yield statement) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads saved pipeline and metrics when the API boots up."""
    global MODEL_PIPELINE, METRICS_DATA

    model_path = Path("artifacts/model.pkl")
    metrics_path = Path("artifacts/metrics.json")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model at: {model_path}, run train_model.py"
        )

    with open(model_path, "rb") as f:
        MODEL_PIPELINE = pickle.load(f)

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            METRICS_DATA = json.load(f)

    yield  # FIX 1: Yield control back to FastAPI while running

    MODEL_PIPELINE = None


app = FastAPI(title="House Price Predictor API", lifespan=lifespan)


# --- 2. Strict Pydantic Schema ---
class PropertyFeatures(BaseModel):
    # FIX 2: Added '=True' to populate_by_name and relaxed strict mode to allow clean numeric conversion
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    first_flr_sf: int = Field(alias="1stFlrSF")
    second_flr_sf: int = Field(alias="2ndFlrSF")
    three_ssn_porch: int = Field(alias="3SsnPorch")
    MSSubClass: int
    MSZoning: str
    LotFrontage: Optional[float] = None
    LotArea: int
    Street: str
    Alley: Optional[str] = None
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: Optional[str] = None
    MasVnrArea: Optional[float] = None
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: Optional[str] = None
    BsmtCond: Optional[str] = None
    BsmtExposure: Optional[str] = None
    BsmtFinType1: Optional[str] = None
    BsmtFinSF1: Optional[float] = None
    BsmtFinType2: Optional[str] = None
    BsmtFinSF2: Optional[float] = None
    BsmtUnfSF: Optional[float] = None
    TotalBsmtSF: Optional[float] = None
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: Optional[str] = None
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: Optional[float] = None
    BsmtHalfBath: Optional[float] = None
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: Optional[str] = None
    GarageType: Optional[str] = None
    GarageYrBlt: Optional[float] = None
    GarageFinish: Optional[str] = None
    GarageCars: Optional[float] = None
    GarageArea: Optional[float] = None
    GarageQual: Optional[str] = None
    GarageCond: Optional[str] = None
    PavedDrive: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    ScreenPorch: int
    PoolArea: int
    PoolQC: Optional[str] = None
    Fence: Optional[str] = None
    MiscFeature: Optional[str] = None
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str


class PredictionResponse(BaseModel):
    predicted_price_usd: float


class BatchPropertyInput(BaseModel):
    items: List[PropertyFeatures]


# --- 3. API Endpoints ---


@app.get("/health")
def health_check():
    """Confirms model readiness."""
    return {
        "status": "online",
        "model_loaded": MODEL_PIPELINE is not None,
        "metrics": METRICS_DATA,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(property_data: PropertyFeatures):
    """Predicts price for a single house."""
    if MODEL_PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    # FIX 3: Fixed double parenthetical syntax in model_dump
    df_input = pd.DataFrame([property_data.model_dump(by_alias=True)])

    log_pred = float(MODEL_PIPELINE.predict(df_input)[0])
    dollar_pred = float(np.expm1(log_pred))

    return PredictionResponse(predicted_price_usd=round(dollar_pred, 2))


@app.post("/predict/batch")
def predict_batch(batch_data: BatchPropertyInput):
    """Predicts prices for a list of houses in one request."""
    if MODEL_PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    df_input = pd.DataFrame(
        [item.model_dump(by_alias=True) for item in batch_data.items]
    )

    log_preds = MODEL_PIPELINE.predict(df_input)
    dollar_preds = np.expm1(log_preds)

    # FIX 4: Corrected zip iteration over predictions
    results = [
        PredictionResponse(predicted_price_usd=round(float(dollar), 2))
        for dollar in dollar_preds
    ]

    return {"predictions": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)