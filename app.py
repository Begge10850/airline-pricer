import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Setup ---
st.set_page_config(page_title="Airline Pricing Advisor", layout="wide")
st.title("✈️ Internal Airline Ticket Pricing Advisor (INR ₹)")
st.markdown("Predict base ticket prices based on route, airline, and time-of-day selections.")

# --- Load Data and Model ---
@st.cache_resource
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
    st.subheader("1. Route")
    source_city = st.selectbox("Source City", [""] + sorted(df['source_city'].unique()))

    if source_city:
        destination_options = sorted(df[df['source_city'] == source_city]['destination_city'].unique())
    else:
        destination_options = []

    destination_city = st.selectbox("Destination City", [""] + destination_options)

# --- Panel 2: Airline and Time Selection ---
with col2:
    st.subheader("2. Airline & Time")

    flight_record = None

    if source_city and destination_city and source_city != destination_city:
        route_df = df[(df['source_city'] == source_city) & (df['destination_city'] == destination_city)]

        airlines = sorted(route_df['airline'].unique())
        airline = st.selectbox("Airline", [""] + airlines)

        if airline:
            airline_df = route_df[route_df['airline'] == airline]
            departure_options = sorted(airline_df['departure_time'].unique())
            departure_time = st.selectbox("Departure Time", [""] + departure_options)

            if departure_time:
                # Filter arrival times based on selected departure time
                matching_df = airline_df[airline_df['departure_time'] == departure_time]
                arrival_options = sorted(matching_df['arrival_time'].unique())
                arrival_time = st.selectbox("Arrival Time", [""] + arrival_options)

                if arrival_time:
                    # Find matching flight record (first match is fine)
                    matched = matching_df[matching_df['arrival_time'] == arrival_time]
                    if not matched.empty:
                        flight_record = matched.iloc[0]
                        st.markdown(f"**Stops:** {flight_record['stops']}")
                    else:
                        st.warning("No matching flight found for selected time combination.")
        else:
            st.info("Select an airline to see available times.")
    elif source_city == destination_city and source_city:
        st.warning("Source and Destination cities cannot be the same.")

# --- Panel 3: Pricing Input ---
with col3:
    st.subheader("3. Pricing Scenario")

    if flight_record is not None:
        class_options = sorted(df['class'].unique())
        default_class = flight_record['class']
        flight_class = st.selectbox("Class", options=class_options, index=class_options.index(default_class))

        days_left = st.slider("Days Left Until Departure", min_value=1, max_value=50, value=15)

        if st.button("🔮 Predict Base Price"):
            input_data = pd.DataFrame({
                'airline': [flight_record['airline']],
                'source_city': [source_city],
                'departure_time': [flight_record['departure_time']],
                'stops': [flight_record['stops']],
                'arrival_time': [flight_record['arrival_time']],
                'destination_city': [destination_city],
                'class': [flight_class.lower()],
                'duration': [flight_record['duration']],  # used internally
                'days_left': [days_left]
            })

            input_processed = preprocessor.transform(input_data)
            predicted_log_price = model.predict(input_processed)
            predicted_price = np.expm1(predicted_log_price)[0]

            st.success(f"### 💰 Predicted Base Ticket Price: ₹{predicted_price:,.2f}")
    else:
        st.info("Select route, airline, and times to enable prediction.")
