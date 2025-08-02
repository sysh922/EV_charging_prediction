
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="EV Charging Demand Prediction", layout="wide")

model = joblib.load('forecasting_ev_model.pkl')

# === Enhanced Theme Styling ===
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
            color: #FFFFFF;
        }
        .main-title {
            font-size: 38px;
            font-weight: 700;
            text-align: center;
            color: #00ffcc;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #e0f7fa;
            margin-bottom: 25px;
        }
        .section-title {
            font-size: 22px;
            color: #80deea;
        }
        .selectbox-label {
            font-size: 18px;
            font-weight: bold;
            color: #b2ebf2;
            margin-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# === Header Section ===
st.markdown("""
    <div class='main-title'>🔋 EV Charging Demand Prediction</div>
    <div class='subtitle'>Predict the future of electric vehicle adoption across counties in Washington State</div>
""", unsafe_allow_html=True)

st.image("ev_charging.jpg", use_container_width=True)

# === Load Data ===
@st.cache_data

def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# === Sidebar for Input ===
st.sidebar.markdown("""<div class='section-title'>🔎 Select County</div>""", unsafe_allow_html=True)
county_list = sorted(df['County'].dropna().unique().tolist())
county = st.sidebar.selectbox("", county_list, key="county_selector")

if county not in df['County'].unique():
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

# === Forecasting Logic ===
historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df['months_since_start'].max()
latest_date = county_df['Date'].max()
forecast_horizon = 36
future_rows = []

for i in range(1, forecast_horizon + 1):
    forecast_date = latest_date + pd.DateOffset(months=i)
    months_since_start += 1
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    recent_cumulative = cumulative_ev[-6:]
    ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

    new_row = {
        'months_since_start': months_since_start,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct_change_1,
        'ev_total_pct_change_3': pct_change_3,
        'ev_growth_slope': ev_growth_slope
    }

    pred = model.predict(pd.DataFrame([new_row]))[0]
    future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

    historical_ev.append(pred)
    if len(historical_ev) > 6:
        historical_ev.pop(0)

    cumulative_ev.append(cumulative_ev[-1] + pred)
    if len(cumulative_ev) > 6:
        cumulative_ev.pop(0)

# === Combine Historical + Forecast ===
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Source'] = 'Historical'
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

forecast_df = pd.DataFrame(future_rows)
forecast_df['Source'] = 'Forecast'
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

combined = pd.concat([
    historical_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# === Plot ===
st.subheader(f"📊 Cumulative EV Forecast for {county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby('Source'):
    ax.plot(data['Date'], data['Cumulative EV'], label=label, marker='o')
ax.set_title(f"Cumulative EV Trend - {county} (3 Years Forecast)", fontsize=14, color='white')
ax.set_xlabel("Date", color='white')
ax.set_ylabel("Cumulative EV Count", color='white')
ax.grid(True, alpha=0.3)
ax.set_facecolor("#1c1c1c")
fig.patch.set_facecolor('#1c1c1c')
ax.tick_params(colors='white')
ax.legend()
st.pyplot(fig)

# === Forecast Summary ===
historical_total = historical_cum['Cumulative EV'].iloc[-1]
forecasted_total = forecast_df['Cumulative EV'].iloc[-1]
if historical_total > 0:
    forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
    st.success(f"EV adoption in **{county}** is projected to **{trend} of {forecast_growth_pct:.2f}%** over the next 3 years.")
else:
    st.warning("Historical EV total is zero; forecast percentage change unavailable.")

# === Multi-County Trend Comparison ===
st.markdown("""<div class='section-title'>📍 EV Adoption Trend Comparison (Multi-County)</div>""", unsafe_allow_html=True)
selected_counties = st.multiselect("Select multiple counties for comparison", county_list, default=[county])

fig, ax = plt.subplots(figsize=(12, 6))
for cnty in selected_counties:
    cnty_df = df[df['County'] == cnty].sort_values("Date")
    cnty_df['Cumulative EV'] = cnty_df['Electric Vehicle (EV) Total'].cumsum()
    ax.plot(cnty_df['Date'], cnty_df['Cumulative EV'], label=cnty)

ax.set_title("Historical Cumulative EV Trend Comparison", fontsize=14, color='white')
ax.set_xlabel("Date", color='white')
ax.set_ylabel("Cumulative EV Count", color='white')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_facecolor("#1c1c1c")
fig.patch.set_facecolor('#1c1c1c')
ax.tick_params(colors='white')
st.pyplot(fig)


# === Footer ===
st.markdown("""
---
<div style='text-align: center; color: #B0BEC5;'>
Prepared for the <b>AICTE Internship Cycle 2 by S4F</b> <br>
By Shivanshu Yadav | Mentor: Raghunandan Sir
</div>
""", unsafe_allow_html=True)
