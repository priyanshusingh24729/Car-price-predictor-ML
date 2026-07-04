"""
app.py
======
Streamlit frontend for the Car Price Prediction project.

Run with:
    streamlit run frontend/app.py

This module ONLY handles UI/UX concerns. All prediction logic lives in
`src/predict.py` (CarPricePredictor), which this app imports and calls
-- keeping the model pipeline fully decoupled from presentation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Allow importing from src/ when running via `streamlit run frontend/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import CarPricePredictor, InvalidInputError  # noqa: E402
from src import config  # noqa: E402

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        font-size: 1rem;
    }
    .predict-btn button {
        background-color: #2E7D32;
        color: white;
        border: none;
    }
    .reset-btn button {
        background-color: #424242;
        color: white;
        border: none;
    }
    .price-card {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1rem;
    }
    .price-card h1 {
        color: white;
        font-size: 2.8rem;
        margin: 0;
    }
    .price-card p {
        color: #c8e6c9;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    .metric-box {
        background-color: #1e2127;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #30343b;
    }
    .warning-box {
        background-color: #3e2723;
        border-left: 4px solid #ff9800;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached resource loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_predictor() -> CarPricePredictor:
    """Load the trained model once per server process, not per request."""
    return CarPricePredictor()


@st.cache_data
def load_company_model_options() -> dict:
    """Load the company -> [models] mapping used to populate dropdowns."""
    options_path = Path(__file__).resolve().parent / "company_model_options.json"
    with open(options_path, "r") as f:
        return json.load(f)


def reset_callback() -> None:
    """Reset all form widgets to their default state."""
    for key in ["company_select", "model_select", "year_input", "kms_input", "fuel_select"]:
        if key in st.session_state:
            del st.session_state[key]


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚗 Car Price Predictor")
    st.markdown(
        "Predicts the resale price of a used car in the Indian market, "
        "trained on real Quikr used-car listings."
    )
    st.divider()

    st.markdown("### 📊 Model Performance")
    st.markdown(
        """
        | Metric | Value |
        |---|---|
        | MAE | ₹1,76,805 |
        | RMSE | ₹3,80,169 |
        | R² (log-scale) | 0.674 |
        | R² (₹ scale) | 0.206 |
        """
    )
    st.caption(
        "⚠️ Predictions are most reliable for mainstream brands "
        "(Maruti, Hyundai, Honda, Tata, Mahindra). Rare luxury brands "
        "(Audi, BMW, Mercedes, Jaguar) have very few training examples "
        "and predictions for them should be treated as rough estimates."
    )

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "Model: Linear Regression (scikit-learn)\n\n"
        "Features: manufacturing year, kilometers driven, brand "
        "(target-encoded), model (target-encoded), fuel type (one-hot)"
    )


# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
st.title("🚗 Used Car Price Predictor")
st.markdown("Fill in the car details below to get an estimated resale price.")
st.markdown("---")

try:
    predictor = load_predictor()
    company_options = load_company_model_options()
except FileNotFoundError:
    st.error(
        "⚠️ Trained model not found. Please run `python main.py --stage train` "
        "before launching the app."
    )
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"⚠️ Failed to load model artifacts: {exc}")
    st.stop()

col_form, col_result = st.columns([1.1, 1], gap="large")

with col_form:
    st.subheader("Car Details")

    company = st.selectbox(
        "Company / Brand",
        options=sorted(company_options.keys()),
        key="company_select",
        help="Manufacturer of the car",
    )

    available_models = company_options.get(company, [])
    model_name = st.selectbox(
        "Model",
        options=available_models,
        key="model_select",
        help="Specific model line for the selected brand",
    )

    year_col, kms_col = st.columns(2)
    with year_col:
        year = st.number_input(
            "Manufacturing Year",
            min_value=config.MIN_VALID_YEAR,
            max_value=config.CURRENT_YEAR,
            value=2015,
            step=1,
            key="year_input",
        )
    with kms_col:
        kms_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=config.MAX_VALID_KMS,
            value=40000,
            step=1000,
            key="kms_input",
        )

    fuel_type = st.selectbox(
        "Fuel Type",
        options=predictor.artifacts.fuel_type_categories,
        key="fuel_select",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_col, reset_col = st.columns(2)
    with predict_col:
        st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
        predict_clicked = st.button("🔮 Predict Price", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
    with reset_col:
        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        st.button("🔄 Reset", on_click=reset_callback)
        st.markdown("</div>", unsafe_allow_html=True)

with col_result:
    st.subheader("Prediction")

    if predict_clicked:
        try:
            result = predictor.predict(
                company=company,
                model=model_name,
                year=int(year),
                kms_driven=int(kms_driven),
                fuel_type=fuel_type,
            )
            st.markdown(
                f"""
                <div class="price-card">
                    <p>Estimated Resale Price</p>
                    <h1>₹{result.predicted_price:,.0f}</h1>
                    <p>{company} {model_name} · {int(year)} · {int(kms_driven):,} km · {fuel_type}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if company in ("Audi", "BMW", "Mercedes", "Jaguar", "Mini", "Volvo", "Land"):
                st.markdown(
                    """
                    <div class="warning-box">
                    ⚠️ <b>Low-confidence estimate</b>: this brand has very few
                    listings in the training data, so the prediction leans
                    heavily on the overall market average and may not reflect
                    true luxury-segment pricing.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except InvalidInputError as exc:
            st.error(f"Invalid input: {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong while predicting: {exc}")
    else:
        st.info("👈 Fill in the car details and click **Predict Price** to see the estimate.")

st.markdown("---")
st.caption(
    "Built with scikit-learn Linear Regression · Data source: Quikr used-car listings (India) · "
    "For educational purposes; not financial advice."
)
