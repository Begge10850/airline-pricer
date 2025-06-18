import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set the page config as the first Streamlit command
st.set_page_config(page_title="Airline Price Advisor", layout="wide")


# --- 1. Load Artifacts and Data ---
# Use cache_data for efficiency
@st.cache_data
def load_data_and_artifacts():
    # Load the dataset that contains all flight details
    df = pd.read_csv('data/cleaned_flight_data.csv')
    # Load the pre-trained model and preprocessor
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('flight_price_model.joblib')
    return df, preprocessor, model

try:
    df, preprocessor, model = load_data_and_artifacts()
except FileNotFoundError:
    st.error("Model, preprocessor, or data files not found. Please ensure all necessary files are present.")
    st.stop()


# --- 2. Build the User Interface (UI) ---

st.title("✈️ Professional Airline Pricing Advisor")
st.write("Select a route and a specific flight to get a dynamic price prediction.")


# --- Create a more realistic, cascading input system ---
col1, col2, col3 = st.columns(3)

# --- Column 1: Route Selection ---
with col1:
    st.header("1. Select Route")
    source = st.selectbox("Source City", options=sorted(df['source_city'].unique()))
    destination = st.selectbox("Destination City", options=sorted(df['destination_city'].unique()))

# --- Column 2: Flight Selection (Dynamically Updated) ---
with col2:
    st.header("2. Select Flight")
    
    # Filter the dataframe based on the selected route
    route_flights = df[(df['source_city'] == source) & (df['destination_city'] == destination)]
    
    # Get the unique flight numbers for that route
    flight_options = sorted(route_flights['flight'].unique())
    
    if flight_options:
        selected_flight = st.selectbox("Flight Number", options=flight_options)
        
        # Get the details for the single selected flight
        flight_details = route_flights[route_flights['flight'] == selected_flight].iloc[0]
        
        # Display the auto-populated details
        st.info(f"**Airline:** {flight_details['airline']}")
        st.info(f"**Stops:** {flight_details['stops']}")
        st.info(f"**Typical Duration:** {flight_details['duration']:.2f} hours")
        st.info(f"**Departure Time:** {flight_details['departure_time']}")
        st.info(f"**Arrival Time:** {flight_details['arrival_time']}")
    else:
        st.warning("No direct flights found for this route in the dataset.")
        st.stop() # Stop the app if no flights are available

# --- Column 3: Dynamic Pricing Inputs ---
with col3:
    st.header("3. Price a Scenario")
    flight_class = st.selectbox("Class", options=sorted(df['class'].unique()))
    days_left = st.slider("Days Left Before Departure", min_value=1, max_value=50, value=20)
    
    predict_button = st.button("Predict Price", type="primary")

# --- Prediction Logic ---
if predict_button:
    # Create the input DataFrame for the model
    # We use the auto-populated flight_details for most columns
    input_data = pd.DataFrame({
        'airline': [flight_details['airline']],
        'source_city': [source],
        'departure_time': [flight_details['departure_time']],
        'stops': [flight_details['stops']],
        'arrival_time': [flight_details['arrival_time']],
        'destination_city': [destination],
        'class': [flight_class.lower()], # Standardize to lowercase
        'duration': [flight_details['duration']],
        'days_left': [days_left]
    })

    # Preprocess the input data
    input_processed = preprocessor.transform(input_data)
    
    # Make a prediction
    predicted_price_log = model.predict(input_processed)
    
    # Inverse transform to get the actual price
    predicted_price = np.expm1(predicted_price_log)[0]

    # Display the result
    st.success(f"Predicted Flight Price:  €{predicted_price:,.2f}")

