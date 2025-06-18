import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set the page config
st.set_page_config(page_title="Airline Price Advisor", layout="wide")

# --- Load Artifacts and Data ---
@st.cache_data
def load_data_and_artifacts():
    df = pd.read_csv('data/Clean_Dataset_EDA_Processed.csv')
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('flight_price_model.joblib')
    return df, preprocessor, model

try:
    df, preprocessor, model = load_data_and_artifacts()
except FileNotFoundError:
    st.error("Model, preprocessor, or data files not found.")
    st.stop()

# --- Title ---
st.title("✈️ Professional Airline Pricing Advisor")
st.write("This tool predicts flight prices based on a machine learning model. Follow the steps below.")

# --- UI Form ---
with st.form("pricing_scenario_form"):
    col1, col2, col3 = st.columns(3)

    # --- Panel 1: Select Route ---
    with col1:
        st.header("1. Select Route")
        city_options = sorted(df['source_city'].unique())
        source_city = st.selectbox("Source City", options=[""] + city_options)

        if source_city:
            valid_destinations = sorted(df[df['source_city'] == source_city]['destination_city'].unique())
        else:
            valid_destinations = []

        destination_city = st.selectbox("Destination City", options=[""] + valid_destinations)

    # --- Panel 2: Select Flight ---
    with col2:
        st.header("2. Select Flight")
        flight_details = None

        if source_city and destination_city:
            if source_city == destination_city:
                st.warning("Source and Destination cities cannot be the same.")
            else:
                route_flights = df[(df['source_city'] == source_city) & (df['destination_city'] == destination_city)]
                flight_options = sorted(route_flights['flight'].unique())

                if flight_options:
                    selected_flight = st.selectbox("Flight Number", options=flight_options)

                    flight_details = route_flights[route_flights['flight'] == selected_flight].iloc[0]
                    st.info(f"**Airline:** {flight_details['airline']}", icon="✈️")
                    st.info(f"**Stops:** {flight_details['stops']}", icon="➡️")
                    st.info(f"**Typical Duration:** {flight_details['duration']:.2f} hours", icon="⏱️")
                else:
                    st.error("No flights found for the selected route.")
        else:
            st.info("Please select both Source and Destination to see flight options.")

    # --- Panel 3: Price a Scenario ---
    with col3:
        st.header("3. Price a Scenario")

        if flight_details is not None:
            class_options = sorted(df['class'].unique())
            default_class_index = class_options.index(flight_details['class'])
            flight_class = st.selectbox("Class", options=class_options, index=default_class_index)
            days_left = st.slider("Days Left Before Departure", min_value=1, max_value=50, value=20)
        else:
            st.info("Please select a valid route and flight to continue.")
            flight_class = None
            days_left = None

    # --- Submit Button (inside the form!) ---
    predict_button = st.form_submit_button("Predict Price", type="primary", use_container_width=True)

# --- Price Prediction ---
if predict_button and flight_details is not None:
    input_data = pd.DataFrame({
        'airline': [flight_details['airline']],
        'source_city': [source_city],
        'departure_time': [flight_details['departure_time']],
        'stops': [flight_details['stops']],
        'arrival_time': [flight_details['arrival_time']],
        'destination_city': [destination_city],
        'class': [flight_class.lower()],
        'duration': [flight_details['duration']],
        'days_left': [days_left]
    })

    input_processed = preprocessor.transform(input_data)
    predicted_price_log = model.predict(input_processed)
    predicted_price = np.expm1(predicted_price_log)[0]

    st.success(f"**Predicted Flight Price:** \n# €{predicted_price:,.2f}")
