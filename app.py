import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Setup ---
st.set_page_config(page_title="Airline Pricing Advisor", layout="wide")
st.title("✈️ Interactive Flight Price Explorer & Advisor")

# --- Load Data and Model ---
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

# --- User Input Filters in the Sidebar ---
st.sidebar.header("Filter Flights")

# 1. Start with blank dropdowns by adding a placeholder
source_city = st.sidebar.selectbox("Source City", ["---"] + sorted(df['source_city'].unique()))

# --- Filtering Logic ---
# Start with the full dataframe and filter it down sequentially
filtered_df = df.copy()

if source_city != "---":
    filtered_df = filtered_df[filtered_df['source_city'] == source_city]
    destination_city = st.sidebar.selectbox("Destination City", ["---"] + sorted(filtered_df['destination_city'].unique()))

    if destination_city != "---":
        if source_city == destination_city:
            st.sidebar.error("Source and Destination must be different.")
        else:
            filtered_df = filtered_df[filtered_df['destination_city'] == destination_city]
            airline = st.sidebar.selectbox("Airline", ["---"] + sorted(filtered_df['airline'].unique()))

            if airline != "---":
                filtered_df = filtered_df[filtered_df['airline'] == airline]
                
                # Smart time filtering
                time_choice = st.sidebar.radio("Filter by Time:", ("Departure", "Arrival"), horizontal=True)
                if time_choice == "Departure":
                    departure_time = st.sidebar.selectbox("Departure Time", ["---"] + sorted(filtered_df['departure_time'].unique()))
                    if departure_time != "---":
                        filtered_df = filtered_df[filtered_df['departure_time'] == departure_time]
                else: # Arrival Time was chosen
                    arrival_time = st.sidebar.selectbox("Arrival Time", ["---"] + sorted(filtered_df['arrival_time'].unique()))
                    if arrival_time != "---":
                        filtered_df = filtered_df[filtered_df['arrival_time'] == arrival_time]

                # Specific days_left filter
                days_left_options = sorted(filtered_df['days_left'].unique())
                days_left_filter = st.sidebar.selectbox("Days Left Until Departure", ["---"] + days_left_options)
                
                if days_left_filter != "---":
                    filtered_df = filtered_df[filtered_df['days_left'] == days_left_filter]


# --- Display Filtered Flights Table ---
# This section will only appear if the dataframe has been filtered
if len(filtered_df) < len(df):
    st.header("Available Flight Options")
    st.write(f"Found {len(filtered_df)} matching flights based on your filters.")
    
    display_cols = ['flight', 'departure_time', 'arrival_time', 'duration', 'stops', 'class', 'price']
    st.dataframe(filtered_df[display_cols])

    # --- Prediction Section ---
    if not filtered_df.empty:
        st.header("Price Prediction for a Scenario")
        st.write("Predicting price based on the **first flight** in the table above. You can adjust the class.")
        
        flight_to_price = filtered_df.iloc[0]
        
        flight_class = st.selectbox("Select Class to Predict", sorted(df['class'].unique()), index=sorted(df['class'].unique()).index(flight_to_price['class']))
        
        if st.button("🔮 Predict Price", type="primary"):
            input_data = pd.DataFrame({
                'airline': [flight_to_price['airline']],
                'source_city': [flight_to_price['source_city']],
                'departure_time': [flight_to_price['departure_time']],
                'stops': [flight_to_price['stops']],
                'arrival_time': [flight_to_price['arrival_time']],
                'destination_city': [flight_to_price['destination_city']],
                'class': [flight_class.lower()], # Use the selected class
                'duration': [flight_to_price['duration']],
                'days_left': [flight_to_price['days_left']]
            })

            input_processed = preprocessor.transform(input_data)
            predicted_log_price = model.predict(input_processed)
            predicted_price = np.expm1(predicted_log_price)[0]

            st.success(f"### 💰 Predicted Base Ticket Price:\n\n₹{predicted_price:,.0f}")
else:
    st.info("ℹ️ Please use the filters in the sidebar to find flights and predict prices.")
