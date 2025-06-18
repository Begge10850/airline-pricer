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
    df = pd.read_csv('data/Clean_Dataset_EDA_Processed.csv')
    # Load the pre-trained model and preprocessor
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('flight_price_model.joblib')
    return df, preprocessor, model

try:
    df, preprocessor, model = load_data_and_artifacts()
except FileNotFoundError:
    st.error("Model, preprocessor, or data files not found. Please ensure all necessary files are present.")
    st.stop()


# --- 2. Application Title ---
st.title("✈️ Professional Airline Pricing Advisor")
st.write("This tool predicts flight prices based on a machine learning model. Follow the steps below.")


# --- 3. Build the User Interface (UI) ---
# We use st.form to group inputs and prevent the app from re-running on every widget change.
with st.form("pricing_scenario_form"):
    
    # Create three columns for our 3-panel layout
    col1, col2, col3 = st.columns(3)

    # --- Panel 1: Select Route ---
    with col1:
        st.header("1. Select Route")
        source_city = st.selectbox("Source City", options=sorted(df['source_city'].unique()))
        
        # Filter available destinations for the selected source
        valid_destinations = sorted(df[df['source_city'] == source_city]['destination_city'].unique())
        destination_city = st.selectbox("Destination City", options=valid_destinations)
        
        if not flight_options:
            st.error(f"No flights found from {source_city} to {destination_city}. Try a different route.")
            selected_flight = None
            flight_details = None


    # --- Panel 2: Select Flight (Dynamically Updated) ---
    with col2:
        st.header("2. Select Flight")
        
        # Filter the dataframe based on the selected route
        # Optional: Filter out same city selection
        if source_city == destination_city:
            st.warning("Source and Destination cities cannot be the same.")
            flight_options = []
        else:
            route_flights = df[(df['source_city'] == source_city) & (df['destination_city'] == destination_city)]
            flight_options = sorted(route_flights['flight'].unique())
        
        # Check if any flights exist for the selected route
        if flight_options:
            selected_flight = st.selectbox("Flight Number", options=flight_options, 
                                           help="Select a flight to see its details. This list updates based on your route selection.")
            
            # Get the details for the single selected flight
            flight_details = route_flights[route_flights['flight'] == selected_flight].iloc[0]
            
            # Display the auto-populated details with tooltips
            st.info(f"**Airline:** {flight_details['airline']}", icon="✈️")
            st.info(f"**Stops:** {flight_details['stops']}", icon="➡️")
            st.info(f"**Typical Duration:** {flight_details['duration']:.2f} hours", icon="⏱️")

        else:
            st.error("No flights found for the selected route in our dataset.")
            selected_flight = None
            flight_details = None 

    # --- Panel 3: Price a Scenario ---
    with col3:
        st.header("3. Price a Scenario")

        if flight_details:
            class_options = sorted(df['class'].unique())
            default_class_index = class_options.index(flight_details['class'])
            flight_class = st.selectbox("Class", options=class_options, index=default_class_index,
                                        help="Price the flight for a specific cabin class.")
            days_left = st.slider("Days Left Before Departure", min_value=1, max_value=50, value=20,
                                help="Simulate the price based on how far in advance the booking is made.")
        else:
            st.info("Please select a valid route and flight to continue.")
            flight_class = None
            days_left = None

    st.write("")
    predict_button = st.form_submit_button("Predict Price", type="primary", use_container_width=True)


# --- 4. Prediction Logic and Display ---
# This block only runs when the "Predict Price" button inside the form is clicked.
if predict_button and flight_details:
    # Create the input DataFrame for the model
    # We use the auto-populated flight_details and the user's scenario inputs
    input_data = pd.DataFrame({
        'airline': [flight_details['airline']],
        'source_city': [source_city],
        'departure_time': [flight_details['departure_time']], # auto-populated
        'stops': [flight_details['stops']],
        'arrival_time': [flight_details['arrival_time']],     # auto-populated
        'destination_city': [destination_city],
        'class': [flight_class.lower()], # User-selected
        'duration': [flight_details['duration']],
        'days_left': [days_left] # User-selected
    })

    # Preprocess the input data
    input_processed = preprocessor.transform(input_data)
    
    # Make a prediction
    predicted_price_log = model.predict(input_processed)
    
    # Inverse transform to get the actual price
    predicted_price = np.expm1(predicted_price_log)[0]

    # Display the result in a clear, formatted way
    st.success(f"**Predicted Flight Price:** \n# €{predicted_price:,.2f}")