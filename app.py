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

# --- Initialize session state defaults ---
for key in ["source_city", "destination_city", "airline", "departure_time", "arrival_time", "flight_class", "days_left", "time_filter_type"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# --- Reset Button ---
if st.button("🔄 Reset Form"):
    for key in ["source_city", "destination_city", "airline", "departure_time", "arrival_time", "flight_class", "days_left", "time_filter_type"]:
        st.session_state[key] = ""
    st.rerun()

# --- Step 1: Route & Airline Selection ---
st.subheader("1️⃣ Select Route and Airline")

col1, col2, col3 = st.columns(3)
source_city = col1.selectbox("Source City", [""] + sorted(df['source_city'].unique()), key="source_city")

if source_city:
    destinations = sorted(df[df['source_city'] == source_city]['destination_city'].unique())
else:
    destinations = []

destination_city = col2.selectbox("Destination City", [""] + destinations, key="destination_city")

if source_city and destination_city:
    airlines = sorted(df[
        (df['source_city'] == source_city) &
        (df['destination_city'] == destination_city)
    ]['airline'].unique())
else:
    airlines = []

airline = col3.selectbox("Airline", [""] + airlines, key="airline")

# --- Step 2: Select Time Type (Departure or Arrival) ---
st.subheader("2️⃣ Select Preferred Time Type")
time_filter_type = st.radio("Filter by:", ["Departure", "Arrival"], horizontal=True, key="time_filter_type")

flight_filter = df[
    (df['source_city'] == source_city) &
    (df['destination_city'] == destination_city) &
    (df['airline'] == airline)
]

col4, col5 = st.columns(2)

departure_time = ""
arrival_time = ""

if time_filter_type == "Departure":
    dep_options = sorted(flight_filter['departure_time'].dropna().unique())
    departure_time = col4.selectbox("Departure Time", [""] + list(dep_options), key="departure_time")
else:
    arr_options = sorted(flight_filter['arrival_time'].dropna().unique())
    arrival_time = col5.selectbox("Arrival Time", [""] + list(arr_options), key="arrival_time")

# --- Step 3: Price Simulation Form ---
st.subheader("3️⃣ Price Simulation")

with st.form("predict_form"):
    col6, col7 = st.columns(2)
    flight_class = col6.selectbox("Class", sorted(df['class'].unique()), key="flight_class")
    days_left = col7.slider("Days Left Until Departure", 1, 50, 20, key="days_left")

    predict_button = st.form_submit_button("🔮 Predict Price")

# --- Prediction Logic ---
if predict_button:
    query = (
        (df['source_city'] == source_city) &
        (df['destination_city'] == destination_city) &
        (df['airline'] == airline)
    )

    if time_filter_type == "Departure" and departure_time:
        query &= (df['departure_time'] == departure_time)
    elif time_filter_type == "Arrival" and arrival_time:
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
