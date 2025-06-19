import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math

# --- Page Setup ---
st.set_page_config(page_title="Airline Pricing Advisor", layout="wide")
st.title("✈️ Smart Airline Ticket Pricing Advisor")
st.markdown("Predict ticket prices based on realistic flight scenarios.")

# --- Load Data and Model ---
@st.cache_data
def load_data_and_artifacts():
    df = pd.read_csv('data/cleaned_flight_data.csv')
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('flight_price_model.joblib')
    return df, preprocessor, model

try:
    df, preprocessor, model = load_data_and_artifacts()
except FileNotFoundError:
    st.error("Required model or data files not found. Please ensure all necessary files are present.")
    st.stop()

# --- UI Columns ---
col1, col2, col3 = st.columns(3)

# --- Panel 1: Route & Airline ---
with col1:
    st.subheader("1. Select Route & Airline")
    source_city = st.selectbox("Source City", sorted(df['source_city'].unique()))
    
    # Filter destinations based on source
    possible_destinations = sorted(df[df['source_city'] == source_city]['destination_city'].unique())
    destination_city = st.selectbox("Destination City", possible_destinations)
    
    # Filter airlines based on the selected route
    route_df = df[(df['source_city'] == source_city) & (df['destination_city'] == destination_city)]
    airline = st.selectbox("Airline", sorted(route_df['airline'].unique()))

# --- Panel 2: Timings & Duration (The New Smart Logic) ---
with col2:
    st.subheader("2. Select Timings & Duration")
    
    # Filter further by selected airline
    flight_options_df = route_df[route_df['airline'] == airline]

    # --- Smart Time Selection ---
    # Allow user to select EITHER departure or arrival time, which then filters the other
    time_choice = st.radio("Select by:", ("Departure Time", "Arrival Time"), horizontal=True)

    if time_choice == "Departure Time":
        departure_time = st.selectbox("Departure Time", sorted(flight_options_df['departure_time'].unique()))
        # Filter arrival times based on the chosen departure time
        possible_arrivals = sorted(flight_options_df[flight_options_df['departure_time'] == departure_time]['arrival_time'].unique())
        arrival_time = st.selectbox("Arrival Time", possible_arrivals, disabled=True) # Display but disable
    else: # Arrival Time was chosen
        arrival_time = st.selectbox("Arrival Time", sorted(flight_options_df['arrival_time'].unique()))
        # Filter departure times based on the chosen arrival time
        possible_departures = sorted(flight_options_df[flight_options_df['arrival_time'] == arrival_time]['departure_time'].unique())
        departure_time = st.selectbox("Departure Time", possible_departures, disabled=True) # Display but disable
    
    # --- Dynamic Duration Slider ---
    # Find the min and max duration for the selected route and airline
    min_duration = flight_options_df['duration'].min()
    max_duration = flight_options_df['duration'].max()

    # The slider now uses these dynamic values
    # We round to make the slider steps cleaner
    duration = st.slider("Flight Duration (in hours)", 
                         min_value=math.floor(min_duration), 
                         max_value=math.ceil(max_duration),
                         value=round(flight_options_df['duration'].mean(), 1),
                         step=0.5)

# --- Panel 3: Pricing Scenario ---
with col3:
    st.subheader("3. Final Scenario")
    
    # We can infer the number of stops from our filtered data
    # We'll take the most common number of stops for this route/airline/time combo
    final_flight_options = flight_options_df[
        (flight_options_df['departure_time'] == departure_time) & 
        (flight_options_df['arrival_time'] == arrival_time)
    ]
    if not final_flight_options.empty:
        stops = final_flight_options['stops'].mode()[0]
        st.info(f"**Inferred Stops:** {stops}", icon="➡️")
    else:
        stops = 0 # Default if no data found
        st.warning("No data for this exact time combo; using 0 stops.")

    flight_class = st.selectbox("Class", sorted(df['class'].unique()))
    days_left = st.slider("Days Left Until Departure", min_value=1, max_value=50, value=15)

    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            'airline': [airline],
            'source_city': [source_city],
            'departure_time': [departure_time],
            'stops': [stops],
            'arrival_time': [arrival_time],
            'destination_city': [destination_city],
            'class': [flight_class.lower()],
            'duration': [duration],
            'days_left': [days_left]
        })

        input_processed = preprocessor.transform(input_data)
        predicted_log_price = model.predict(input_processed)
        predicted_price = np.expm1(predicted_log_price)[0]

        st.success(f"### 💰 Predicted Base Ticket Price:\n\n₹{predicted_price:,.0f}")
