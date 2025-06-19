import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Setup ---
st.set_page_config(page_title="Airline Pricing Advisor", layout="wide")
st.title("✈️ Internal Airline Ticket Pricing Advisor (INR ₹)")

# --- Load Data and Model ---
@st.cache_resource
def load_data_and_artifacts():
    df = pd.read_csv("data/Clean_Dataset_EDA_Processed.csv")
    preprocessor = joblib.load("preprocessor.joblib")
    model = joblib.load("flight_price_model.joblib")
    return df, preprocessor, model

df, preprocessor, model = load_data_and_artifacts()

# --- Initialize Session State ---
defaults = {
    "source_city": "",
    "destination_city": "",
    "airline": "",
    "time_filter_type": "Departure",
    "departure_time": "",
    "arrival_time": "",
    "flight_class": "",
    "days_left": 15,
    "submitted": False
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Reset Button ---
if st.button("🔄 Reset Form"):
    for key in defaults:
        st.session_state[key] = defaults[key]

# --- Inputs UI ---
st.subheader("Fill the details below and press 'Predict Price'")

col1, col2, col3 = st.columns(3)

with col1:
    st.session_state.source_city = st.selectbox("Source City", [""] + sorted(df['source_city'].unique()), index=0)

with col2:
    if st.session_state.source_city:
        destinations = sorted(df[df['source_city'] == st.session_state.source_city]['destination_city'].unique())
    else:
        destinations = []
    st.session_state.destination_city = st.selectbox("Destination City", [""] + destinations, index=0)

with col3:
    if st.session_state.source_city and st.session_state.destination_city and st.session_state.source_city != st.session_state.destination_city:
        airlines = sorted(df[
            (df['source_city'] == st.session_state.source_city) &
            (df['destination_city'] == st.session_state.destination_city)
        ]['airline'].unique())
    else:
        airlines = []
    st.session_state.airline = st.selectbox("Airline", [""] + airlines, index=0)

st.divider()

# Time Filter Choice
st.session_state.time_filter_type = st.radio("Select Time Filter:", ["Departure", "Arrival"], horizontal=True)

# Time options
time_df = df[
    (df['source_city'] == st.session_state.source_city) &
    (df['destination_city'] == st.session_state.destination_city) &
    (df['airline'] == st.session_state.airline)
]

col4, col5 = st.columns(2)

with col4:
    if st.session_state.time_filter_type == "Departure":
        options = sorted(time_df['departure_time'].unique())
        st.session_state.departure_time = st.selectbox("Departure Time", [""] + options, index=0)
    else:
        st.session_state.departure_time = ""

with col5:
    if st.session_state.time_filter_type == "Arrival":
        options = sorted(time_df['arrival_time'].unique())
        st.session_state.arrival_time = st.selectbox("Arrival Time", [""] + options, index=0)
    else:
        st.session_state.arrival_time = ""

st.divider()

col6, col7 = st.columns(2)

with col6:
    st.session_state.flight_class = st.selectbox("Class", sorted(df['class'].unique()))

with col7:
    st.session_state.days_left = st.slider("Days Left Until Departure", min_value=1, max_value=50, value=st.session_state.days_left)

# --- Submit Button ---
if st.button("🔮 Predict Price"):
    st.session_state.submitted = True

# --- Prediction ---
if st.session_state.submitted:
    # Validate required fields
    if (
        st.session_state.source_city and
        st.session_state.destination_city and
        st.session_state.airline and
        ((st.session_state.time_filter_type == "Departure" and st.session_state.departure_time) or
         (st.session_state.time_filter_type == "Arrival" and st.session_state.arrival_time))
    ):
        query = (
            (df['source_city'] == st.session_state.source_city) &
            (df['destination_city'] == st.session_state.destination_city) &
            (df['airline'] == st.session_state.airline)
        )

        if st.session_state.time_filter_type == "Departure":
            query &= (df['departure_time'] == st.session_state.departure_time)
        else:
            query &= (df['arrival_time'] == st.session_state.arrival_time)

        matched = df[query]

        if not matched.empty:
            record = matched.iloc[0]

            input_df = pd.DataFrame({
                'airline': [record['airline']],
                'source_city': [record['source_city']],
                'departure_time': [record['departure_time']],
                'stops': [record['stops']],
                'arrival_time': [record['arrival_time']],
                'destination_city': [record['destination_city']],
                'class': [st.session_state.flight_class.lower()],
                'duration': [record['duration']],
                'days_left': [st.session_state.days_left]
            })

            input_processed = preprocessor.transform(input_df)
            predicted_log_price = model.predict(input_processed)
            predicted_price = np.expm1(predicted_log_price)[0]

            st.success(f"### 💰 Predicted Base Ticket Price: ₹{predicted_price:,.0f}")
        else:
            st.error("❌ No matching flight found for this combination.")
    else:
        st.warning("⚠️ Please fill all fields before predicting.")
