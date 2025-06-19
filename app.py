import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Setup ---
st.set_page_config(page_title="Airline Pricing Advisor", layout="wide")
st.title("✈️ Interactive Airline Pricing Advisor")
st.markdown("Use the filters to explore available flights and predict ticket prices.")

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

# --- User Input and Filtering Logic ---
st.sidebar.header("Filter Flights")

# 1. Empty Dropdown Menus
# By adding a placeholder like "---" at the beginning of the options list,
# the dropdown will default to a blank selection.
source_city = st.sidebar.selectbox("Source City", ["---"] + sorted(df['source_city'].unique()))

if source_city != "---":
    destination_city = st.sidebar.selectbox("Destination City", ["---"] + sorted(df[df['source_city'] == source_city]['destination_city'].unique()))
else:
    destination_city = st.sidebar.selectbox("Destination City", [""], disabled=True)

# Start with the full dataframe and filter it down based on user selections
filtered_df = df.copy()

if source_city != "---" and destination_city != "---":
    if source_city == destination_city:
        st.error("Source and Destination cities cannot be the same.")
        st.stop()
    else:
        # Filter by route
        filtered_df = df[(df['source_city'] == source_city) & (df['destination_city'] == destination_city)]

        airline = st.sidebar.selectbox("Airline", ["---"] + sorted(filtered_df['airline'].unique()))
        
        if airline != "---":
            # Filter by airline
            filtered_df = filtered_df[filtered_df['airline'] == airline]

            # 2. Smart Time Selection
            time_choice = st.sidebar.radio("Filter by:", ("Departure Time", "Arrival Time"), horizontal=True)

            if time_choice == "Departure Time":
                departure_time = st.sidebar.selectbox("Departure Time", ["---"] + sorted(filtered_df['departure_time'].unique()))
                if departure_time != "---":
                    # Filter by selected departure time
                    filtered_df = filtered_df[filtered_df['departure_time'] == departure_time]
            else: # Arrival Time was chosen
                arrival_time = st.sidebar.selectbox("Arrival Time", ["---"] + sorted(filtered_df['arrival_time'].unique()))
                if arrival_time != "---":
                    # Filter by selected arrival time
                    filtered_df = filtered_df[filtered_df['arrival_time'] == arrival_time]

# --- Display Filtered Flights Table ---
st.header("Available Flights")
st.write(f"Showing {len(filtered_df)} matching flights.")

# We only display the most relevant columns for the user
display_cols = ['flight', 'airline', 'departure_time', 'arrival_time', 'duration', 'stops', 'class', 'price', 'days_left']
st.dataframe(filtered_df[display_cols], height=300)

# --- Prediction Section ---
st.header("Price Prediction for a Scenario")

# We can use the first flight from the filtered table as a default for prediction
if not filtered_df.empty:
    st.write("Predicting price based on the first flight in the table above. You can adjust the Class and Days Left.")
    
    # Use the details from the first row of the filtered table
    flight_to_price = filtered_df.iloc[0]
    
    col1, col2 = st.columns(2)
    with col1:
        flight_class = st.selectbox("Class", sorted(df['class'].unique()), index=sorted(df['class'].unique()).index(flight_to_price['class']))
    with col2:
        days_left = st.slider("Days Left Until Departure", min_value=1, max_value=50, value=int(flight_to_price['days_left']))

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
            'days_left': [days_left] # Use the selected days_left
        })

        input_processed = preprocessor.transform(input_data)
        predicted_log_price = model.predict(input_processed)
        predicted_price = np.expm1(predicted_log_price)[0]

        st.success(f"### 💰 Predicted Base Ticket Price:\n\n₹{predicted_price:,.0f}")
else:
    st.info("Please select a valid route and airline to see available flights and make a prediction.")

