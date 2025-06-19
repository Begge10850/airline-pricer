import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Setup ---
st.set_page_config(page_title="Airline Pricing Advisor", layout="wide")
st.title("✈️ Internal Airline Ticket Pricing Advisor (INR ₹)")

# --- Load Data and Model ---
@st.cache_resource
def load_data_and_artifacts():
    df = pd.read_csv("data/Clean_Dataset_EDA_Processed.csv")
    preprocessor = joblib.load("preprocessor.joblib")
    model = joblib.load("flight_price_model.joblib")
    return df, preprocessor, model

df, preprocessor, model = load_data_and_artifacts()

# --- Initialize Session State ---
if "predict_triggered" not in st.session_state:
    st.session_state.predict_triggered = False

if "reset" not in st.session_state:
    st.session_state.reset = False

# --- Refresh Button ---
if st.button("🔄 Reset Form"):
    st.session_state.clear()
    st.rerun()

# --- Input Fields ---
with st.form("input_form"):
    st.subheader("✈️ Fill in Flight Details")

    col1, col2, col3 = st.columns(3)
    source_city = col1.selectbox("Source City", [""] + sorted(df['source_city'].unique()))

    destination_city = ""
    if source_city:
        destinations = sorted(df[df['source_city'] == source_city]['destination_city'].unique())
        destination_city = col2.selectbox("Destination City", [""] + destinations)
    else:
        destination_city = col2.selectbox("Destination City", [""])

    airline = ""
    if source_city and destination_city and source_city != destination_city:
        airlines = sorted(df[
            (df['source_city'] == source_city) &
            (df['destination_city'] == destination_city)
        ]['airline'].unique())
        airline = col3.selectbox("Airline", [""] + airlines)
    else:
        airline = col3.selectbox("Airline", [""])

    st.divider()

    time_filter_type = st.radio("Choose Time Filter:", ["Departure", "Arrival"], horizontal=True)

    time_df = df[
        (df['source_city'] == source_city) &
        (df['destination_city'] == destination_city) &
        (df['airline'] == airline)
    ]

    col4, col5 = st.columns(2)

    departure_time = ""
    arrival_time = ""

    if time_filter_type == "Departure":
        departure_options = sorted(time_df['departure_time'].unique())
        departure_time = col4.selectbox("Departure Time", [""] + departure_options)
    else:
        arrival_options = sorted(time_df['arrival_time'].unique())
        arrival_time = col5.selectbox("Arrival Time", [""] + arrival_options)

    col6, col7 = st.columns(2)
    flight_class = col6.selectbox("Class", sorted(df['class'].unique()))
    days_left = col7.slider("Days Left Until Departure", min_value=1, max_value=50, value=15)

    # Submit Button
    predict_button = st.form_submit_button("🔮 Predict Price")
    if predict_button:
        st.session_state.predict_triggered = True
        st.session_state.prediction_inputs = {
            "source_city": source_city,
            "destination_city": destination_city,
            "airline": airline,
            "departure_time": departure_time if time_filter_type == "Departure" else "",
            "arrival_time": arrival_time if time_filter_type == "Arrival" else "",
            "class": flight_class,
            "days_left": days_left
        }

# --- Prediction ---
if st.session_state.get("predict_triggered"):
    inputs = st.session_state.prediction_inputs
    query = (
        (df['source_city'] == inputs['source_city']) &
        (df['destination_city'] == inputs['destination_city']) &
        (df['airline'] == inputs['airline'])
    )
    if time_filter_type == "Departure":
        query &= (df['departure_time'] == inputs['departure_time'])
    else:
        query &= (df['arrival_time'] == inputs['arrival_time'])

    matched = df[query]

    if not matched.empty:
        record = matched.iloc[0]

        input_df = pd.DataFrame({
            'airline': [record['airline']],
            'source_city': [record['source_city']],
            'departure_time': [record['departure_time']],
            'stops': [record['stops']],
            'arrival_time': [record['arrival_time']],
            'destination_city': [record['destination_city']],
            'class': [inputs['class'].lower()],
            'duration': [record['duration']],
            'days_left': [inputs['days_left']]
        })

        input_processed = preprocessor.transform(input_df)
        predicted_log_price = model.predict(input_processed)
        predicted_price = np.expm1(predicted_log_price)[0]

        st.success(f"### 💰 Predicted Base Ticket Price: ₹{predicted_price:,.0f}")
    else:
        st.error("❌ No matching flight found for this combination.")
