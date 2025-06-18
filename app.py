import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set the page config as the first Streamlit command
st.set_page_config(page_title="Airline Price Advisor", layout="wide")


# --- 1. Load the Saved Model and Preprocessor ---

@st.cache_data
def load_artifacts():
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('flight_price_model.joblib')
    return preprocessor, model

try:
    preprocessor, model = load_artifacts()
except FileNotFoundError:
    st.error("Model or preprocessor files not found. Please run the modeling notebook first.")
    st.stop()


# --- 2. Initialize Session State ---
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
if 'show_prediction' not in st.session_state:
    st.session_state['show_prediction'] = False


# --- 3. Create the User Interface (UI) ---

st.title("✈️ Dynamic Airline Price Advisor")
st.write("Enter the details of the flight to get a price prediction.")

with st.form("prediction_form"):
    
    airline_options = ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India']
    source_city_options = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
    departure_time_options = ['Evening', 'Early_Morning', 'Morning', 'Afternoon', 'Night', 'Late_Night']
    stops_options = [0, 1, 2]
    arrival_time_options = ['Night', 'Morning', 'Early_Morning', 'Afternoon', 'Evening', 'Late_Night']
    destination_city_options = ['Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi']
    class_options = ['Economy', 'Business']

    col1, col2 = st.columns(2)

    with col1:
        st.header("Flight Details")
        airline = st.selectbox("Airline", options=airline_options, key='airline')
        source_city = st.selectbox("Source City", options=source_city_options, key='source_city')
        destination_city = st.selectbox("Destination City", options=destination_city_options, key='destination_city')
        stops = st.selectbox("Number of Stops", options=stops_options, key='stops')
        flight_class = st.selectbox("Class", options=class_options, key='flight_class')

    with col2:
        st.header("Timing & Duration")
        departure_time = st.selectbox("Departure Time", options=departure_time_options, key='departure_time')
        arrival_time = st.selectbox("Arrival Time", options=arrival_time_options, key='arrival_time')
        duration = st.number_input("Duration (in hours)", min_value=0.5, max_value=50.0, value=2.5, step=0.5, key='duration')
        days_left = st.slider("Days Left Before Departure", min_value=1, max_value=50, value=20, key='days_left')

    submit_col, clear_col = st.columns([1, 0.2])

    with submit_col:
        submitted = st.form_submit_button("Predict Price")

    with clear_col:
        cleared = st.form_submit_button("Clear")


# --- 4. Prediction and Clear Logic ---

if submitted:
    st.session_state['show_prediction'] = True
    
    # Create a DataFrame from the user's input
    input_data = pd.DataFrame({
        'airline': [st.session_state.airline],
        'source_city': [st.session_state.source_city],
        'departure_time': [st.session_state.departure_time], # <-- THE FIX IS HERE
        'stops': [st.session_state.stops],
        'arrival_time': [st.session_state.arrival_time],
        'destination_city': [st.session_state.destination_city],
        'class': [st.session_state.flight_class.lower()],
        'duration': [st.session_state.duration],
        'days_left': [st.session_state.days_left]
    })

    input_processed = preprocessor.transform(input_data)
    predicted_price_log = model.predict(input_processed)
    predicted_price = np.expm1(predicted_price_log)[0]
    
    st.session_state['prediction'] = predicted_price

if cleared:
    st.session_state['show_prediction'] = False
    st.session_state['prediction'] = None


# --- 5. Display the Result ---

if st.session_state['show_prediction'] and st.session_state['prediction'] is not None:
    st.success(f"Predicted Flight Price:  €{st.session_state['prediction']:,.2f}")

