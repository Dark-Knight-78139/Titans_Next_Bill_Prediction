"""
Titans Next Bill Prediction - Streamlit UI
===========================================
Run from the repository root:

    pip install streamlit
    streamlit run app.py

Tabs
----
1. Single prediction  - enter the 15 model inputs (or load an existing customer record)
2. Batch prediction   - upload a CSV, or score the latest billing cycle in the bundled dataset
3. Customer history   - bill trend for one customer plus the model's next-cycle forecast
4. Model insights     - model details, feature importance, test-set metrics vs. a naive baseline
5. Retrain            - re-run the repo's ingestion -> transformation -> training pipeline
"""
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths / setup (the repo's pipeline uses paths relative to the repo root)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pickles may have been created with a slightly different scikit-learn version.
warnings.filterwarnings("ignore", message=".*Trying to unpickle.*")

MODEL_PATH = ROOT / "artifacts" / "model.pkl"
PREPROCESSOR_PATH = ROOT / "artifacts" / "preprocessor.pkl"
TEST_PATH = ROOT / "artifacts" / "test.csv"
DATA_PATHS = [ROOT / "notebook" / "data" / "Next_Predict_Bill.csv", ROOT / "artifacts" / "data.csv"]

TARGET = "target_next_bill"
CURRENCY = ""  # set to e.g. "₹" or "$" if you want a symbol in front of amounts

# The exact numeric columns used in src/components/data_transformation.py
NUMERIC_FEATURES = [
    "Monthly rental",
    "is_unlimited_data",
    "is_unlimited_calls",
    "Extra data",
    "Extra Call min",
    "est_isd_total_cost",
    "data_utilization_ratio",
    "call_utilization_ratio",
    "Current Bill Amount",
    "prev_bill_t1",
    "bill_mom_change",
    "rolling_3m_bill_mean",
    "rolling_3m_data_mean",
    "expanding_cust_bill_mean",
    "expanding_cust_bill_std",
]
DEFAULT_PLANS = [399, 599, 999, 1499]

DEFAULTS = {
    "Monthly rental": 599,
    "is_unlimited_data": False,
    "is_unlimited_calls": False,
    "Extra data": 0.0,
    "Extra Call min": 0,
    "est_isd_total_cost": 0.0,
    "data_utilization_ratio": 0.85,
    "call_utilization_ratio": 0.65,
    "Current Bill Amount": 1000.0,
    "prev_bill_t1": 950.0,
    "bill_mom_change": 50.0,
    "rolling_3m_bill_mean": 980.0,
    "rolling_3m_data_mean": 75.0,
    "expanding_cust_bill_mean": 970.0,
    "expanding_cust_bill_std": 135.0,
}


def fmt(x: float) -> str:
    return f"{CURRENCY}{x:,.2f}"


def key(name: str) -> str:
    return f"in_{name}"


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    """Return (model, preprocessor) or raise if artifacts are missing/unreadable."""
    from src.utils import load_object  # repo helper (dill)

    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("artifacts/model.pkl or artifacts/preprocessor.pkl not found.")
    return load_object(str(MODEL_PATH)), load_object(str(PREPROCESSOR_PATH))


@st.cache_data(show_spinner=False)
def load_data():
    for p in DATA_PATHS:
        if p.exists():
            return pd.read_csv(p)
    return None


def artifact_stamp() -> tuple:
    """Changes whenever the model/preprocessor files change (used as a cache key)."""
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in (MODEL_PATH, PREPROCESSOR_PATH))


def get_artifacts_or_warn():
    try:
        return load_artifacts()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load model artifacts: {e}")
        st.info("Use the **Retrain** tab to build `model.pkl` and `preprocessor.pkl`.")
        return None


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
def to_model_frame(df: pd.DataFrame, preprocessor) -> pd.DataFrame:
    """Align a dataframe to the columns the fitted preprocessor expects."""
    cols = list(getattr(preprocessor, "feature_names_in_", NUMERIC_FEATURES))
    return df.reindex(columns=cols)


def predict_df(df: pd.DataFrame, model, preprocessor) -> np.ndarray:
    X = preprocessor.transform(to_model_frame(df, preprocessor))
    return model.predict(X)


def ensemble_spread(model, preprocessor, row_df: pd.DataFrame):
    """10th-90th percentile of the individual estimators' predictions (rough spread, not a CI)."""
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        return None
    try:
        X = preprocessor.transform(to_model_frame(row_df, preprocessor))
        preds = np.array([e.predict(X)[0] for e in estimators])
        return float(np.percentile(preds, 10)), float(np.percentile(preds, 90))
    except Exception:  # noqa: BLE001
        return None


def validate_batch(df: pd.DataFrame):
    """Fill bill_mom_change if possible and report missing required columns."""
    df = df.copy()
    if "bill_mom_change" not in df.columns and {"Current Bill Amount", "prev_bill_t1"} <= set(df.columns):
        df["bill_mom_change"] = df["Current Bill Amount"] - df["prev_bill_t1"]
    missing = [c for c in NUMERIC_FEATURES if c not in df.columns]
    return df, missing


# --------------------------------------------------------------------------- #
# Tab 1: Single prediction
# --------------------------------------------------------------------------- #
def init_inputs():
    for name, default in DEFAULTS.items():
        st.session_state.setdefault(key(name), default)
    st.session_state.setdefault("auto_mom", True)


def _apply_record():
    """on_click callback: copy a dataset row into the input widgets."""
    df = load_data()
    cid, date = st.session_state["load_cust"], st.session_state["load_date"]
    row = df[(df["Customer_id"] == cid) & (df["Billing Date"] == date)].iloc[0]
    for name in NUMERIC_FEATURES:
        v = row[name]
        if name in ("is_unlimited_data", "is_unlimited_calls"):
            st.session_state[key(name)] = bool(int(v))
        elif name in ("Monthly rental", "Extra Call min"):
            st.session_state[key(name)] = int(v)
        else:
            st.session_state[key(name)] = float(v)


def load_record_ui():
    df = load_data()
    if df is None or "Customer_id" not in df.columns:
        return
    with st.expander("Load an existing customer record (optional)"):
        c1, c2, c3 = st.columns([1, 1, 1])
        customers = sorted(df["Customer_id"].unique().tolist())
        cid = c1.selectbox("Customer ID", customers, key="load_cust")
        dates = sorted(df.loc[df["Customer_id"] == cid, "Billing Date"].unique().tolist())
        # default to the latest billing date
        if st.session_state.get("load_date") not in dates:
            st.session_state["load_date"] = dates[-1]
        c2.selectbox("Billing date", dates, key="load_date")
        c3.write("")
        c3.write("")
        c3.button("Fill inputs from this record", on_click=_apply_record)


def single_prediction_tab():
    init_inputs()
    load_record_ui()

    data = load_data()
    plans = sorted(data["Monthly rental"].unique().tolist()) if data is not None else DEFAULT_PLANS
    if st.session_state[key("Monthly rental")] not in plans:
        plans = sorted(set(plans) | {st.session_state[key("Monthly rental")]})

    st.subheader("Plan & usage (current cycle)")
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Monthly rental (plan)", plans, key=key("Monthly rental"))
    c2.toggle("Unlimited data", key=key("is_unlimited_data"))
    c3.toggle("Unlimited calls", key=key("is_unlimited_calls"))

    c1, c2, c3 = st.columns(3)
    c1.number_input("Extra data (beyond plan)", min_value=0.0, step=0.5, key=key("Extra data"))
    c2.number_input("Extra call minutes", min_value=0, step=1, key=key("Extra Call min"))
    c3.number_input("Estimated ISD total cost", min_value=0.0, step=10.0, key=key("est_isd_total_cost"))

    c1, c2 = st.columns(2)
    c1.number_input("Data utilization ratio", min_value=0.0, max_value=5.0, step=0.01,
                    key=key("data_utilization_ratio"), help="Data used / data allowance. >1 means over the allowance.")
    c2.number_input("Call utilization ratio", min_value=0.0, max_value=5.0, step=0.01,
                    key=key("call_utilization_ratio"), help="Call minutes used / allowance. >1 means over the allowance.")

    st.subheader("Billing history")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Current bill amount", min_value=0.0, step=10.0, key=key("Current Bill Amount"))
    c2.number_input("Previous bill (t-1)", min_value=0.0, step=10.0, key=key("prev_bill_t1"))
    auto = c3.checkbox("Auto-calculate month-over-month change", key="auto_mom",
                       help="bill_mom_change = current bill - previous bill (matches the dataset).")
    if auto:  # must be set before the widget below is created
        st.session_state[key("bill_mom_change")] = round(
            st.session_state[key("Current Bill Amount")] - st.session_state[key("prev_bill_t1")], 2)
    c3.number_input("Bill change vs previous month", step=10.0, key=key("bill_mom_change"), disabled=auto)

    c1, c2 = st.columns(2)
    c1.number_input("Rolling 3-month bill mean", min_value=0.0, step=10.0, key=key("rolling_3m_bill_mean"))
    c2.number_input("Rolling 3-month data mean", min_value=0.0, step=1.0, key=key("rolling_3m_data_mean"))
    c1, c2 = st.columns(2)
    c1.number_input("Customer's lifetime avg bill (expanding mean)", min_value=0.0, step=10.0,
                    key=key("expanding_cust_bill_mean"))
    c2.number_input("Customer's bill std. deviation (expanding)", min_value=0.0, step=1.0,
                    key=key("expanding_cust_bill_std"))

    if st.button("Predict next bill", type="primary"):
        artifacts = get_artifacts_or_warn()
        if artifacts is None:
            return
        model, pre = artifacts
        values = {n: st.session_state[key(n)] for n in NUMERIC_FEATURES}
        for n in ("is_unlimited_data", "is_unlimited_calls"):
            values[n] = int(values[n])
        row = pd.DataFrame([values])
        pred = float(predict_df(row, model, pre)[0])
        cur = values["Current Bill Amount"]

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted next bill", fmt(pred))
        m2.metric("Current bill", fmt(cur), delta=f"{pred - cur:+,.2f} expected", delta_color="inverse")
        spread = ensemble_spread(model, pre, row)
        if spread:
            m3.metric("Ensemble spread (P10-P90)", f"{fmt(spread[0])} - {fmt(spread[1])}",
                      help="Spread of the individual trees' predictions. A rough indication only, not a confidence interval.")
        st.bar_chart(pd.Series({
            "Previous bill": values["prev_bill_t1"],
            "Current bill": cur,
            "3-month avg": values["rolling_3m_bill_mean"],
            "Predicted next": pred,
        }, name="Amount"))


# --------------------------------------------------------------------------- #
# Tab 2: Batch prediction
# --------------------------------------------------------------------------- #
def batch_tab():
    artifacts = get_artifacts_or_warn()
    if artifacts is None:
        return
    model, pre = artifacts
    data = load_data()

    source = st.radio("Input source", ["Upload CSV", "Latest billing cycle in bundled dataset"], horizontal=True)
    df = None
    if source == "Upload CSV":
        if data is not None:
            template = data[NUMERIC_FEATURES].head(3).copy()
            template.insert(0, "Customer_id", data["Customer_id"].head(3).values)
            st.download_button("Download CSV template", template.to_csv(index=False), "batch_template.csv", "text/csv")
        st.caption("Required columns: " + ", ".join(NUMERIC_FEATURES) +
                   ". `bill_mom_change` is computed automatically if missing. `Customer_id` is optional.")
        up = st.file_uploader("Upload a CSV", type="csv")
        if up is not None:
            df = pd.read_csv(up)
    elif data is None:
        st.warning("Bundled dataset not found.")
    else:
        latest = data["Billing Date"].max()
        df = data[data["Billing Date"] == latest].reset_index(drop=True)
        st.caption(f"Scoring {len(df)} customers for billing date {latest}.")

    if df is None:
        return

    df, missing = validate_batch(df)
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return

    preds = predict_df(df, model, pre)
    out = pd.DataFrame()
    if "Customer_id" in df.columns:
        out["Customer_id"] = df["Customer_id"]
    out["Current Bill Amount"] = df["Current Bill Amount"].values
    out["Predicted next bill"] = np.round(preds, 2)
    out["Expected change"] = (out["Predicted next bill"] - out["Current Bill Amount"]).round(2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers scored", f"{len(out):,}")
    c2.metric("Total predicted", fmt(out["Predicted next bill"].sum()))
    c3.metric("Average predicted", fmt(out["Predicted next bill"].mean()))

    if TARGET in df.columns and df[TARGET].notna().any():
        from sklearn.metrics import mean_absolute_error, r2_score
        ok = df[TARGET].notna().values
        st.info(f"Labels found in file - R²: {r2_score(df[TARGET][ok], preds[ok]):.3f}, "
                f"MAE: {mean_absolute_error(df[TARGET][ok], preds[ok]):,.2f}")

    st.dataframe(out, hide_index=True)
    st.download_button("Download predictions", out.to_csv(index=False), "predicted_next_bill.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Tab 3: Customer history
# --------------------------------------------------------------------------- #
def history_tab():
    data = load_data()
    if data is None or "Customer_id" not in data.columns:
        st.warning("Dataset with Customer_id not found.")
        return
    cid = st.selectbox("Customer", sorted(data["Customer_id"].unique().tolist()), key="hist_cust")
    cust = data[data["Customer_id"] == cid].sort_values("Billing Date").copy()
    cust["Billing Date"] = pd.to_datetime(cust["Billing Date"])

    chart = pd.DataFrame({"Actual bill": cust["Current Bill Amount"].values}, index=cust["Billing Date"])
    chart["Predicted next bill"] = np.nan

    try:
        model, pre = load_artifacts()
        last = cust.tail(1)
        pred = float(predict_df(last, model, pre)[0])
        next_date = last["Billing Date"].iloc[0] + pd.DateOffset(months=1)
        # connect the line: start the forecast series at the last actual point
        chart.iloc[-1, chart.columns.get_loc("Predicted next bill")] = float(last["Current Bill Amount"].iloc[0])
        chart.loc[next_date] = [np.nan, pred]
        c1, c2 = st.columns(2)
        c1.metric("Latest bill", fmt(float(last["Current Bill Amount"].iloc[0])))
        c2.metric(f"Forecast for {next_date:%b %Y}", fmt(pred))
    except Exception:  # noqa: BLE001
        st.caption("Model not available - showing history only.")

    st.line_chart(chart)
    with st.expander("Show records"):
        st.dataframe(cust, hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 4: Model insights
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Evaluating on the test split...")
def evaluate_on_test(stamp: tuple):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if not TEST_PATH.exists():
        return None
    model, pre = load_artifacts()
    test = pd.read_csv(TEST_PATH).dropna(subset=[TARGET])  # same filtering as data_transformation.py
    y = test[TARGET].values
    pred = predict_df(test, model, pre)
    naive = test["Current Bill Amount"].values  # baseline: next bill = current bill
    return {
        "n": len(test),
        "metrics": pd.DataFrame({
            "R²": [r2_score(y, pred), r2_score(y, naive)],
            "MAE": [mean_absolute_error(y, pred), mean_absolute_error(y, naive)],
            "RMSE": [mean_squared_error(y, pred) ** 0.5, mean_squared_error(y, naive) ** 0.5],
        }, index=["Trained model", "Baseline (next = current)"]),
        "scatter": pd.DataFrame({"Actual": y, "Predicted": pred}),
    }


def feature_names(pre):
    try:
        return [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    except Exception:  # noqa: BLE001
        return NUMERIC_FEATURES


def insights_tab():
    artifacts = get_artifacts_or_warn()
    if artifacts is None:
        return
    model, pre = artifacts

    st.subheader("Model")
    st.write(f"**Type:** `{type(model).__name__}`")
    st.json({k: str(v) for k, v in model.get_params().items()}, expanded=False)

    if hasattr(model, "feature_importances_"):
        names = feature_names(pre)
        imp = np.asarray(model.feature_importances_)
        if len(names) == len(imp):
            st.subheader("Feature importance")
            st.bar_chart(pd.Series(imp, index=names, name="importance").sort_values(ascending=False))

    st.subheader("Test-set performance")
    res = evaluate_on_test(artifact_stamp())
    if res is None:
        st.caption("artifacts/test.csv not found.")
        return
    st.caption(f"{res['n']} test rows (rows without a target are dropped, as in training).")
    st.dataframe(res["metrics"].style.format("{:,.3f}"))
    st.scatter_chart(res["scatter"], x="Actual", y="Predicted")


# --------------------------------------------------------------------------- #
# Tab 5: Retrain
# --------------------------------------------------------------------------- #
def retrain_tab():
    st.warning(
        "This re-runs the repo's full pipeline (ingestion -> transformation -> model selection with GridSearchCV "
        "over 8 models) and **overwrites** `artifacts/model.pkl` and `artifacts/preprocessor.pkl`. "
        "It can take several minutes and needs `catboost` and `xgboost` installed. "
        "Existing artifacts are backed up first."
    )
    confirm = st.checkbox("I understand and want to retrain")
    if not st.button("Start retraining", disabled=not confirm):
        return

    with st.spinner("Training... this may take a while"):
        try:
            if MODEL_PATH.exists() or PREPROCESSOR_PATH.exists():
                backup = ROOT / "artifacts" / f"backup_{datetime.now():%Y%m%d_%H%M%S}"
                backup.mkdir(parents=True, exist_ok=True)
                for p in (MODEL_PATH, PREPROCESSOR_PATH):
                    if p.exists():
                        shutil.copy2(p, backup / p.name)

            from src.components.data_ingestion import DataIngestion
            from src.components.data_transformation import DataTransformation
            from src.components.model_trainer import ModelTrainer

            train_path, test_path = DataIngestion().initiate_data_config()
            train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(train_path, test_path)
            best_model, r2 = ModelTrainer().inititate_model_trainer(train_arr, test_arr)
        except Exception as e:  # noqa: BLE001
            st.error(f"Training failed: {e}")
            return

    st.cache_resource.clear()
    st.cache_data.clear()
    st.success(f"Done. Best model: **{type(best_model).__name__}** - test R² = {r2:.3f}. Artifacts reloaded.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Next Bill Predictor", page_icon="📱", layout="wide")
    st.title("📱 Next Bill Predictor")
    st.caption("Predict a telecom customer's bill for the next billing cycle.")

    tabs = st.tabs(["Single prediction", "Batch prediction", "Customer history", "Model insights", "Retrain"])
    with tabs[0]:
        single_prediction_tab()
    with tabs[1]:
        batch_tab()
    with tabs[2]:
        history_tab()
    with tabs[3]:
        insights_tab()
    with tabs[4]:
        retrain_tab()


if __name__ == "__main__":
    main()