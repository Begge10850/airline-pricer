import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Airline Price Advisor", layout="wide")

# Load data & models
@st.cache_data
def load_data_and_artifacts():
    df = pd.read_csv('data/Clean_Dataset_EDA_Processed.csv')
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('flight_price_model.joblib')
    return df, preprocessor, model

try:
    df, preprocessor, model = load_data_and_artifacts()
except FileNotFoundError:
    st.error("Required model or dataset files not found.")
    st.stop()

# App title
st.title("✈️ Internal Airline Ticket Pricing Advisor (INR ₹)")
st.markdown("Use this tool to simulate ticket prices based on real flight details.")

# UI columns
col1, col2, col3 = st.columns(3)

# --- Panel 1: Select Route ---
with col1:
    st.subheader("1. Route Selection")
    source_city = st.selectbox("Source City", options=[""] + sorted(df['source_city'].unique()))

    if source_city:
        destination_options = sorted(df[df['source_city'] == source_city]['destination_city'].unique())
    else:
        destination_options = []

    destination_city = st.selectbox("Destination City", options=[""] + destination_options)

# --- Panel 2: View Airlines & Select Flight ---
with col2:
    st.subheader("2. Flights on this Route")

    flight_details = None
    selected_flight = None

    if source_city and destination_city and source_city != destination_city:
        route_flights = df[(df['source_city'] == source_city) & (df['destination_city'] == destination_city)]
        available_airlines = sorted(route_flights['airline'].unique())
        st.markdown("**Airlines serving this route:**")
        st.write(", ".join(available_airlines))

        flight_options = sorted(route_flights['flight'].unique())
        if flight_options:
            selected_flight = st.selectbox("Flight Number", options=flight_options)
            if selected_flight:
                flight_details = route_flights[route_flights['flight'] == selected_flight].iloc[0]

                st.markdown(f"**Stops:** {flight_details['stops']}")
                st.markdown(f"**Duration:** {flight_details['duration']} hours")
                st.markdown(f"**Departure Time:** {flight_details['departure_time']}")
                st.markdown(f"**Arrival Time:** {flight_details['arrival_time']}")
        else:
            st.warning("No flights found for this route.")
    elif source_city == destination_city and source_city:
        st.warning("Source and destination cities cannot be the same.")

# --- Panel 3: Pricing Inputs ---
with col3:
    st.subheader("3. Price This Flight")

    if flight_details is not None:
        class_options = sorted(df['class'].unique())
        default_class_index = class_options.index(flight_details['class'])
        flight_class = st.selectbox("Class", options=class_options, index=default_class_index)

        days_left = st.slider("Days Left Until Departure", min_value=1, max_value=50, value=20)

        if st.button("🔮 Predict Ticket Price"):
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

            st.success(f"### 💰 Predicted Ticket Price: ₹{predicted_price:,.2f}")
    else:
        st.info("Select a flight to enable price prediction.")
