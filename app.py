import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Setup ---
st.set_page_config(page_title="Airline Price Advisor", layout="wide")
st.title("✈️ Internal Airline Ticket Pricing Advisor (INR ₹)")

# --- Load Data and Model ---
@st.cache_resource
def load_data_and_artifacts():
    df = pd.read_csv("data/Clean_Dataset_EDA_Processed.csv")
    preprocessor = joblib.load("preprocessor.joblib")
    model = joblib.load("flight_price_model.joblib")
    return df, preprocessor, model

df, preprocessor, model = load_data_and_artifacts()

# --- Initialize session state defaults safely ---
defaults = {
    "source_city": "",
    "destination_city": "",
    "airline": "",
    "departure_time": "",
    "arrival_time": "",
    "flight_class": "",
    "days_left_slider": 20,
    "duration_slider": None,
    "time_filter_type": "Departure",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Reset Button ---
if st.button("🔄 Reset Form"):
    for key, value in defaults.items():
        st.session_state[key] = value
    st.rerun()

# --- UI LAYOUT ---
col1, col2, col3 = st.columns(3)

# --- SOURCE CITY ---
st.session_state.source_city = col1.selectbox(
    "Source City",
    [""] + sorted(df['source_city'].unique()),
    index=0,
    key="source_city"
)

# --- DESTINATION CITY ---
if st.session_state.source_city:
    destinations = df[df['source_city'] == st.session_state.source_city]['destination_city'].unique()
else:
    destinations = []

st.session_state.destination_city = col2.selectbox(
    "Destination City",
    [""] + sorted(destinations),
    index=0,
    key="destination_city"
)

# --- TIME FILTER ---
st.session_state.time_filter_type = col3.radio(
    "Filter by:",
    ["Departure", "Arrival"],
    horizontal=True,
    key="time_filter_type"
)

# --- TIME OPTIONS ---
time_col = 'departure_time' if st.session_state.time_filter_type == 'Departure' else 'arrival_time'
if st.session_state.source_city and st.session_state.destination_city:
    df_filtered = df[
        (df['source_city'] == st.session_state.source_city) &
        (df['destination_city'] == st.session_state.destination_city)
    ]
    time_options = df_filtered[time_col].unique()
else:
    df_filtered = pd.DataFrame()
    time_options = []

if st.session_state.time_filter_type == "Departure":
    st.session_state.departure_time = col1.selectbox("Departure Time", [""] + sorted(time_options), key="departure_time")
else:
    st.session_state.arrival_time = col1.selectbox("Arrival Time", [""] + sorted(time_options), key="arrival_time")

# --- AIRLINE ---
if not df_filtered.empty:
    airline_filter = df_filtered[df_filtered[time_col] == st.session_state[time_col]]
    airline_options = airline_filter['airline'].unique()
else:
    airline_options = []

st.session_state.airline = col2.selectbox("Airline", [""] + sorted(airline_options), key="airline")

# --- CLASS & DAYS LEFT ---
col6, col7 = st.columns(2)

with col6:
    st.session_state.flight_class = st.selectbox("Class", sorted(df['class'].unique()), key="flight_class")

with col7:
    days_left_value = st.slider(
        "Days Left Until Departure",
        min_value=1,
        max_value=50,
        value=st.session_state.days_left_slider,
        step=1,
        key="days_left_slider"
    )
    st.session_state.days_left_slider = days_left_value

# --- DYNAMIC DURATION SLIDER ---
if st.session_state.source_city and st.session_state.destination_city and st.session_state.airline:
    duration_range_df = df[
        (df['source_city'] == st.session_state.source_city) &
        (df['destination_city'] == st.session_state.destination_city) &
        (df['airline'] == st.session_state.airline)
    ]

    if not duration_range_df.empty:
        min_duration = float(duration_range_df['duration'].min())
        max_duration = float(duration_range_df['duration'].max())
        default_duration = float(duration_range_df['duration'].median())

        if st.session_state.duration_slider is None:
            st.session_state.duration_slider = round(default_duration, 1)

        selected_duration = st.slider(
            "Typical Duration (Hours)",
            min_value=round(min_duration, 1),
            max_value=round(max_duration, 1),
            value=st.session_state.duration_slider,
            step=0.1,
            key="duration_slider"
        )
        st.session_state.duration_slider = selected_duration

# --- PREDICT BUTTON ---
predict_button = st.button("📊 Predict Price")

# --- PREDICTION LOGIC ---
if predict_button:
    if st.session_state.source_city and st.session_state.destination_city and st.session_state.airline and \
       (st.session_state.departure_time or st.session_state.arrival_time):

        selected_time = st.session_state.departure_time if st.session_state.time_filter_type == "Departure" else st.session_state.arrival_time
        filtered_row = df_filtered[
            (df_filtered['airline'] == st.session_state.airline) &
            (df_filtered[time_col] == selected_time)
        ]

        if not filtered_row.empty:
            flight_row = filtered_row.iloc[0]
            input_data = pd.DataFrame({
                'airline': [st.session_state.airline],
                'source_city': [st.session_state.source_city],
                'departure_time': [flight_row['departure_time']],
                'stops': [flight_row['stops']],
                'arrival_time': [flight_row['arrival_time']],
                'destination_city': [st.session_state.destination_city],
                'class': [st.session_state.flight_class.lower()],
                'duration': [st.session_state.duration_slider],
                'days_left': [st.session_state.days_left_slider],
            })

            input_processed = preprocessor.transform(input_data)
            predicted_price_log = model.predict(input_processed)
            predicted_price = np.expm1(predicted_price_log)[0]

            st.success(f"**Predicted Base Ticket Price:** \n# ₹{predicted_price:,.2f}")
        else:
            st.error("No matching flight found with selected values.")
    else:
        st.warning("Please complete all selections before predicting.")
