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
    df = pd.read_csv('data/Clean_Dataset_EDA_Processed.csv')
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('flight_price_model.joblib')
    return df, preprocessor, model

df, preprocessor, model = load_data_and_artifacts()

# --- Session State Init ---
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# --- Main Form ---
with st.form("prediction_form"):
    st.subheader("Flight Pricing Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        source_city = st.selectbox("Source City", [""] + sorted(df['source_city'].unique()))
    with col2:
        if source_city:
            valid_destinations = sorted(df[df['source_city'] == source_city]['destination_city'].unique())
        else:
            valid_destinations = []
        destination_city = st.selectbox("Destination City", [""] + valid_destinations)
    with col3:
        if source_city and destination_city and source_city != destination_city:
            route_df = df[(df['source_city'] == source_city) & (df['destination_city'] == destination_city)]
            airlines = sorted(route_df['airline'].unique())
        else:
            airlines = []
        airline = st.selectbox("Airline", [""] + airlines)

    col4, col5 = st.columns(2)
    with col4:
        if airline:
            dep_times = sorted(route_df[route_df['airline'] == airline]['departure_time'].unique())
        else:
            dep_times = []
        departure_time = st.selectbox("Departure Time", [""] + dep_times)

    with col5:
        if airline and departure_time:
            arr_times = sorted(route_df[(route_df['airline'] == airline) & (route_df['departure_time'] == departure_time)]['arrival_time'].unique())
        else:
            arr_times = []
        arrival_time = st.selectbox("Arrival Time", [""] + arr_times)

    col6, col7 = st.columns(2)
    with col6:
        class_choice = st.selectbox("Class", options=sorted(df['class'].unique()))
    with col7:
        days_left = st.slider("Days Left Until Departure", min_value=1, max_value=50, value=15)

    # Submit Button
    submitted = st.form_submit_button("🔮 Predict Price")
    if submitted:
        st.session_state.submitted = True
        st.session_state.input_data = {
            "source_city": source_city,
            "destination_city": destination_city,
            "airline": airline,
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "class": class_choice,
            "days_left": days_left
        }

# --- Reset Button ---
if st.button("🔄 Reset Form"):
    st.session_state.submitted = False
    st.session_state.input_data = {}

# --- Prediction Logic ---
if st.session_state.get("submitted"):
    input_values = st.session_state.input_data
    # Match exact flight from data
    match = df[
        (df['source_city'] == input_values['source_city']) &
        (df['destination_city'] == input_values['destination_city']) &
        (df['airline'] == input_values['airline']) &
        (df['departure_time'] == input_values['departure_time']) &
        (df['arrival_time'] == input_values['arrival_time'])
    ]

    if not match.empty:
        record = match.iloc[0]
        input_df = pd.DataFrame({
            'airline': [record['airline']],
            'source_city': [record['source_city']],
            'departure_time': [record['departure_time']],
            'stops': [record['stops']],
            'arrival_time': [record['arrival_time']],
            'destination_city': [record['destination_city']],
            'class': [input_values['class'].lower()],
            'duration': [record['duration']],
            'days_left': [input_values['days_left']]
        })

        input_processed = preprocessor.transform(input_df)
        predicted_log_price = model.predict(input_processed)
        predicted_price = np.expm1(predicted_log_price)[0]

        st.success(f"### 💰 Predicted Base Ticket Price: ₹{predicted_price:,.0f}")
    else:
        st.error("⚠️ No matching flight found for this combination.")
