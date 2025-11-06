import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
from tensorflow.keras.models import load_model
import joblib
from prophet.serialize import model_from_json
import os
import matplotlib.pyplot as plt
from functools import reduce

# -------------------------------------------------------
# --- CONFIG ---
# -------------------------------------------------------
API_KEY = "af885a4715dd3eb9ec36ed84e7de199a5f2e894573f0e0bd643c42e865017bd3"
BASE_DIR = r"D:\Coding_folder\Main_Project\Chatbot\Langgraph_Chatbot\Model"

# --- Fixed Sensors ---
POLLUTANTS = {
    "PM2.5 (µg/m³)": 12234735,
    "PM10 (µg/m³)": 12234734,
    "Ozone (µg/m³)": 12234733
}

FEATURES = {
    "RH (%)": 12234736,
    "WS (m/s)": 14341693,
    "WD (deg)": 14341691
}

MODEL_PATHS = {
    "PM2.5 (µg/m³)": os.path.join(BASE_DIR, "PM2_5_g_m_"),
    "PM10 (µg/m³)": os.path.join(BASE_DIR, "PM10_g_m_"),
    "Ozone (µg/m³)": os.path.join(BASE_DIR, "Ozone_g_m_")
}

# -------------------------------------------------------
# --- Fetch Daily Data ---
# -------------------------------------------------------
def fetch_daily(sensor_id):
    """Fetch last 24 days of daily average readings for a given sensor."""
    datetime_to = datetime.utcnow().isoformat() + "Z"
    datetime_from = (datetime.utcnow() - timedelta(days=24)).isoformat() + "Z"
    url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements/daily"
    headers = {"X-API-Key": API_KEY}
    params = {
        "datetime_from": datetime_from,
        "datetime_to": datetime_to,
        "limit": 100,
        "page": 1
    }

    try:
        res = requests.get(url, headers=headers, params=params, timeout=15).json()
        results = res.get("results", [])
    except Exception:
        return pd.DataFrame()

    data = []
    for r in results:
        date_utc = r.get("period", {}).get("datetimeFrom", {}).get("utc")
        if date_utc:
            dt = datetime.fromisoformat(date_utc.replace("Z", "+00:00"))
            ist = dt.astimezone(pytz.timezone("Asia/Kolkata"))
            # Remove timezone info for Prophet compatibility
            date_local = datetime(ist.year, ist.month, ist.day, 0, 0, 0)
            value = r.get("value", None)
            data.append({"Timestamp": date_local, "value": value})

    df = pd.DataFrame(data)
    return df.drop_duplicates(subset="Timestamp")

# -------------------------------------------------------
# --- Load Prophet + LSTM Models ---
# -------------------------------------------------------
@st.cache_resource
def load_models(pollutant_name):
    path = MODEL_PATHS[pollutant_name]
    base_name = path.split(os.sep)[-1][:-1]
    lstm = load_model(os.path.join(path, f"lstm_model_{base_name}_.h5"), compile=False)
    scaler = joblib.load(os.path.join(path, f"scaler_{base_name}_.pkl"))
    with open(os.path.join(path, f"prophet_model_{base_name}_.json"), "r") as f:
        prophet = model_from_json(f.read())
    return lstm, scaler, prophet

# -------------------------------------------------------
# --- Hybrid Prophet + LSTM Forecast ---
# -------------------------------------------------------
def predict_next_day(user_df, pollutant_name, feature_cols):
    user_df = user_df.sort_values("Timestamp").copy()
    # Remove timezone info for Prophet compatibility
    user_df["Timestamp"] = pd.to_datetime(user_df["Timestamp"]).dt.tz_localize(None)

    # Load models
    lstm_model, scaler, prophet_model = load_models(pollutant_name)

    # Prophet Forecast
    prophet_input = user_df[["Timestamp", pollutant_name]].rename(
        columns={"Timestamp": "ds", pollutant_name: "y"}
    )
    forecast = prophet_model.predict(prophet_input)
    user_df["prophet_yhat"] = forecast["yhat"]

    # Residuals
    user_df["residual"] = user_df[pollutant_name] - user_df["prophet_yhat"]

    # ✅ Use only residual + fixed weather features
    fixed_features = ["residual", "RH (%)", "WS (m/s)", "WD (deg)"]

    # Ensure all exist in the DataFrame
    for col in fixed_features:
        if col not in user_df.columns:
            user_df[col] = 0.0

    # Handle missing values gracefully
    user_df[fixed_features] = user_df[fixed_features].interpolate(limit_direction="both").fillna(
        user_df[fixed_features].mean()
    )

    # --- LSTM Prediction ---
    X = user_df[fixed_features].values
    X_scaled = scaler.transform(X)
    X_scaled = np.expand_dims(X_scaled, axis=0)
    lstm_res_scaled = lstm_model.predict(X_scaled, verbose=0)

    # Inverse-transform residual back to original scale
    lstm_res = scaler.inverse_transform(
        np.hstack([lstm_res_scaled, np.zeros((lstm_res_scaled.shape[0], len(fixed_features) - 1))])
    )[:, 0][0]

    # --- Next Day Forecast ---
    next_day = user_df["Timestamp"].max() + pd.Timedelta(days=1)
    prophet_next = prophet_model.predict(pd.DataFrame({"ds": [next_day]}))["yhat"].values[0]
    hybrid_next = prophet_next + lstm_res

    result = pd.DataFrame({
        "Timestamp": [next_day],
        "Prophet Forecast": [prophet_next],
        "LSTM Residual": [lstm_res],
        "Hybrid Forecast (Final)": [hybrid_next]
    })

    return result, user_df

# -------------------------------------------------------
# --- Streamlit Interface ---
# -------------------------------------------------------
st.title("🌤 Air Pollutants Forecast Dashboard for Delhi")

pollutant = st.selectbox("Select pollutant to predict:", list(POLLUTANTS.keys()))
days_to_predict = st.number_input("Days to predict (max 7):", min_value=1, max_value=7, value=1)

if st.button("Fetch Data & Forecast"):
    st.info("⏳ Fetching pollutant and weather data...")

    dfs = []

    # Fetch pollutant data
    for name, sid in POLLUTANTS.items():
        df = fetch_daily(sid)
        if not df.empty:
            df = df.rename(columns={"value": name})
            dfs.append(df)
        else:
            st.warning(f"No data for pollutant {name}")

    # Fetch weather feature data
    for name, sid in FEATURES.items():
        df = fetch_daily(sid)
        if not df.empty:
            df = df.rename(columns={"value": name})
            dfs.append(df)
        else:
            st.warning(f"No data for feature {name}")

    if not dfs:
        st.error("❌ No data retrieved from OpenAQ.")
        st.stop()

    # Merge all available dataframes
    df_merged = reduce(lambda l, r: pd.merge(l, r, on="Timestamp", how="outer"), dfs)
    df_merged = df_merged.sort_values("Timestamp").reset_index(drop=True)
    df_merged["Timestamp"] = pd.to_datetime(df_merged["Timestamp"]).dt.tz_localize(None)
    df_merged = df_merged.interpolate(limit_direction="both").fillna(method="bfill").fillna(method="ffill")

    st.subheader(" Last 24 Days Data from OpenAQ")
    st.dataframe(df_merged)

    # Feature columns (everything except Timestamp and target pollutant)
    feature_cols = [c for c in df_merged.columns if c not in ["Timestamp", pollutant]]

    # --- Forecast Iteratively ---
    st.info("🔮 Running Hybrid Forecast...")
    forecast_results = []
    temp_df = df_merged.copy()

    for _ in range(days_to_predict):
        result, temp_df = predict_next_day(temp_df, pollutant, feature_cols)
        forecast_results.append(result)

        # Append prediction as new row for next iteration
        new_row = temp_df.iloc[[-1]].copy()
        new_row[pollutant] = result["Hybrid Forecast (Final)"].iloc[0]
        new_row["Timestamp"] = result["Timestamp"].iloc[0]
        temp_df = pd.concat([temp_df, new_row], ignore_index=True)

    df_forecast = pd.concat(forecast_results).reset_index(drop=True)

    # --- Display Results ---
    st.subheader(f"📈 {days_to_predict}-Day Forecast for {pollutant}")
    st.dataframe(df_forecast)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(df_merged["Timestamp"], df_merged[pollutant], label="Actual", marker="o")
    plt.scatter(df_forecast["Timestamp"], df_forecast["Hybrid Forecast (Final)"],
                color="red", label="Hybrid Forecast", s=80)
    plt.xlabel("Date")
    plt.ylabel(pollutant)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title(f"{pollutant} Forecast ({days_to_predict} Days Ahead)")
    st.pyplot(plt)

    # -------------------------------------------------------
    # --- Display Saved Insight Plots from Model Directory ---
    # -------------------------------------------------------
    st.subheader("📊 Model Insight Plots")

    pollutant_dir = MODEL_PATHS[pollutant]  # e.g. ...\PM2_5_g_m_\
    base_name = os.path.basename(os.path.normpath(pollutant_dir))  # e.g. PM2_5_g_m_

    # Exact filenames that match your folder
    prophet_plot = os.path.join(pollutant_dir, f"prophet_components_{base_name}.png")
    hybrid_plot = os.path.join(pollutant_dir, f"hybrid_forecast_{base_name}.png")

    cols = st.columns(2)

    with cols[0]:
        if os.path.exists(prophet_plot):
            st.image(prophet_plot, caption=f"📈 Prophet Components ({pollutant})", use_container_width=True)
        else:
            st.warning(f"⚠️ Prophet components plot not found at:\n{prophet_plot}")

    with cols[1]:
        if os.path.exists(hybrid_plot):
            st.image(hybrid_plot, caption=f"🔮 Hybrid Forecast Plot ({pollutant})", use_container_width=True)
        else:
            st.warning(f"⚠️ Hybrid forecast plot not found at:\n{hybrid_plot}")
