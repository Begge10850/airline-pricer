import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Setup ---
st.set_page_config(page_title="Airline Price Advisor", layout="wide")
st.title("✈️ Internal Airline Ticket Pricing Advisor (INR ₹)")

# --- Load Data and Model ---
@st.cache_resource
def load_data_and_artifacts():
    df = pd.read_csv("data/Clean_Dataset_EDA_Processed.csv")
    preprocessor = joblib.load("preprocessor.joblib")
    model = joblib.load("flight_price_model.joblib")
    return df, preprocessor, model

df, preprocessor, model = load_data_and_artifacts()

# --- Session State Initialization ---
if "predict_triggered" not in st.session_state:
    st.session_state.predict_triggered = False

# --- Refresh Button ---
if st.button("🔄 Reset Form"):
    st.session_state.clear()
    st.rerun()

# --- Step 1: Dynamic Inputs (Outside Form) ---
st.subheader("1️⃣ Select Route and Flight Info")

col1, col2, col3 = st.columns(3)
source_city = col1.selectbox("Source City", [""] + sorted(df['source_city'].unique()))

if source_city:
    destinations = sorted(df[df['source_city'] == source_city]['destination_city'].unique())
else:
    destinations = []

destination_city = col2.selectbox("Destination City", [""] + destinations)

if source_city and destination_city:
    airlines = sorted(df[
        (df['source_city'] == source_city) &
        (df['destination_city'] == destination_city)
    ]['airline'].unique())
else:
    airlines = []

airline = col3.selectbox("Airline", [""] + airlines)

# --- Step 2: Select Time Type (Outside Form) ---
st.subheader("2️⃣ Choose Departure or Arrival Time")
time_filter_type = st.radio("Filter by:", ["Departure", "Arrival"], horizontal=True)

flight_filter = df[
    (df['source_city'] == source_city) &
    (df['destination_city'] == destination_city) &
    (df['airline'] == airline)
]

col4, col5 = st.columns(2)
departure_time = ""
arrival_time = ""

if time_filter_type == "Departure":
    options = sorted(flight_filter['departure_time'].unique())
    departure_time = col4.selectbox("Departure Time", [""] + options)
else:
    options = sorted(flight_filter['arrival_time'].unique())
    arrival_time = col5.selectbox("Arrival Time", [""] + options)

# --- Step 3: Remaining Inputs + Predict Button ---
st.subheader("3️⃣ Price Simulation")

with st.form("predict_form"):
    col6, col7 = st.columns(2)
    flight_class = col6.selectbox("Class", sorted(df['class'].unique()))
    days_left = col7.slider("Days Left Until Departure", min_value=1, max_value=50, value=20)

    predict_button = st.form_submit_button("🔮 Predict Price")

# --- Prediction Logic ---
if predict_button:
    query = (
        (df['source_city'] == source_city) &
        (df['destination_city'] == destination_city) &
        (df['airline'] == airline)
    )

    if time_filter_type == "Departure":
        query &= (df['departure_time'] == departure_time)
    else:
        query &= (df['arrival_time'] == arrival_time)

    matched = df[query]

    if matched.empty:
        st.error("❌ No matching flight found.")
    else:
        record = matched.iloc[0]

        input_df = pd.DataFrame({
            'airline': [record['airline']],
            'source_city': [record['source_city']],
            'departure_time': [record['departure_time']],
            'stops': [record['stops']],
            'arrival_time': [record['arrival_time']],
            'destination_city': [record['destination_city']],
            'class': [flight_class.lower()],
            'duration': [record['duration']],
            'days_left': [days_left]
        })

        input_processed = preprocessor.transform(input_df)
        predicted_log_price = model.predict(input_processed)
        predicted_price = np.expm1(predicted_log_price)[0]

        st.success(f"### 💰 Predicted Base Price: ₹{predicted_price:,.0f}")
