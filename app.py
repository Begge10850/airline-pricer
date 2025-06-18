import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Setup ---
st.set_page_config(page_title="Airline Pricing Advisor", layout="wide")
st.title("✈️ Internal Airline Ticket Pricing Advisor (INR ₹)")
st.markdown("Predict base ticket prices based on selected route, airline, and flight details.")

# --- Load Data and Artifacts ---
@st.cache_data
def load_data_and_artifacts():
    df = pd.read_csv('data/Clean_Dataset_EDA_Processed.csv')
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('flight_price_model.joblib')
    return df, preprocessor, model

try:
    df, preprocessor, model = load_data_and_artifacts()
except FileNotFoundError:
    st.error("Required model or data files not found.")
    st.stop()

# --- UI Columns ---
col1, col2, col3 = st.columns(3)

# --- Panel 1: Route Selection ---
with col1:
    st.subheader("1. Select Route")
    source_city = st.selectbox("Source City", [""] + sorted(df['source_city'].unique()))

    if source_city:
        destination_options = sorted(df[df['source_city'] == source_city]['destination_city'].unique())
    else:
        destination_options = []

    destination_city = st.selectbox("Destination City", [""] + destination_options)

# --- Panel 2: Airline and Flight ---
with col2:
    st.subheader("2. Select Flight")

    airline = None
    flight_details = None

    if source_city and destination_city and source_city != destination_city:
        route_df = df[(df['source_city'] == source_city) & (df['destination_city'] == destination_city)]

        # Step 1: Select Airline
        airlines = sorted(route_df['airline'].unique())
        airline = st.selectbox("Airline", [""] + airlines)

        if airline:
            airline_flights = route_df[route_df['airline'] == airline]
            flight_numbers = sorted(airline_flights['flight'].unique())

            selected_flight = st.selectbox("Flight Number", [""] + flight_numbers)

            if selected_flight:
                flight_details = airline_flights[airline_flights['flight'] == selected_flight].iloc[0]

                # Display fixed flight details
                st.markdown(f"**Stops:** {flight_details['stops']}")
                st.markdown(f"**Duration:** {flight_details['duration']} hours")
                st.markdown(f"**Departure Time:** {flight_details['departure_time']}")
                st.markdown(f"**Arrival Time:** {flight_details['arrival_time']}")
        else:
            st.info("Select an airline to see its flights for this route.")
    elif source_city == destination_city and source_city:
        st.warning("Source and Destination cities cannot be the same.")

# --- Panel 3: Pricing Scenario ---
with col3:
    st.subheader("3. Pricing Input")

    if flight_details is not None:
        class_options = sorted(df['class'].unique())
        default_class = flight_details['class']
        flight_class = st.selectbox("Class", options=class_options, index=class_options.index(default_class))

        days_left = st.slider("Days Left Until Departure", min_value=1, max_value=50, value=15)

        if st.button("🔮 Predict Base Price"):
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

            st.success(f"### 💰 Predicted Base Ticket Price: ₹{predicted_price:,.2f}")
    else:
        st.info("Select airline and flight to enable prediction.")
