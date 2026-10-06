from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import ForecastRequest, ForecastResponse


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_forecasting_model.joblib"
)

STATIC_DIR = (
    PROJECT_ROOT
    / "app"
    / "static"
)

TEMPLATES_DIR = (
    PROJECT_ROOT
    / "app"
    / "templates"
)

EXAMPLES_DIR = (
    PROJECT_ROOT
    / "examples"
)


# ============================================================
# MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


model = joblib.load(MODEL_PATH)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Revenue Demand Forecasting API",
    description="Store sales demand forecasting using machine learning",
    version="1.0.0",
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=STATIC_DIR
    ),
    name="static",
)


# ============================================================
# SAMPLE INPUT FILES
# ============================================================

app.mount(
    "/examples",
    StaticFiles(
        directory=EXAMPLES_DIR
    ),
    name="examples",
)


# ============================================================
# TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory=TEMPLATES_DIR
)


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=ForecastResponse
)
def predict(
    request: ForecastRequest
):

    try:

        input_data = pd.DataFrame(
            [request.model_dump()]
        )

        prediction = model.predict(
            input_data
        )

        predicted_sales = max(
            float(prediction[0]),
            0.0
        )

        return ForecastResponse(
            predicted_sales=predicted_sales
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )