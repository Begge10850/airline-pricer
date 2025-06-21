import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import openai
import os
from dotenv import load_dotenv

# --- Setup ---
st.set_page_config(page_title="Airline Pricing Advisor", layout="wide")
st.title("✈️ Dynamic Pricing & Revenue Advisor")
st.markdown("Predict base ticket prices and receive an optimized recommendation to maximize revenue.")

# --- Load OpenAI Key ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load Data and Model ---
@st.cache_resource
def load_data_and_artifacts():
    df = pd.read_csv("data/Clean_Dataset_EDA_Processed.csv")
    preprocessor = joblib.load("preprocessor.joblib")
    model = joblib.load("flight_price_model.joblib")
    return df, preprocessor, model

try:
    df, preprocessor, model = load_data_and_artifacts()
except FileNotFoundError:
    st.error("Model or dataset not found. Please ensure files are in the correct directory.")
    st.stop()

# --- Optimizer Function ---
def find_optimal_price(base_price, elasticity_factor=1.5, price_range_pct=0.25):
    best_price = base_price
    max_revenue = 0
    base_demand = 100
    base_revenue = base_price * base_demand
    price_range = np.linspace(base_price * (1 - price_range_pct), base_price * (1 + price_range_pct), 100)

    for price in price_range:
        price_diff_percent = (price - base_price) / base_price
        demand_factor = 1 - (price_diff_percent * elasticity_factor)
        demand_at_price = max(0, base_demand * demand_factor)
        expected_revenue = price * demand_at_price
        if expected_revenue > max_revenue:
            max_revenue = expected_revenue
            best_price = price

    uplift = ((max_revenue - base_revenue) / base_revenue) * 100 if base_revenue > 0 else 0
    return {"optimized_price": best_price, "uplift_percent": uplift}

# --- Session State Defaults ---
defaults = {
    "source_city": "", "destination_city": "", "airline": "",
    "time_filter_type": "Departure", "departure_time": "", "arrival_time": "",
    "flight_class": "", "days_left": 15, "submitted": False, 
    "prediction_results": None
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.button("🔄 Reset Form"):
    for key, value in defaults.items():
        st.session_state[key] = value

# --- Input UI ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Route")
    st.session_state.source_city = st.selectbox("Source City", [""] + sorted(df['source_city'].unique()))
    if st.session_state.source_city:
        filtered = df[df['source_city'] == st.session_state.source_city]
        st.session_state.destination_city = st.selectbox("Destination City", [""] + sorted(filtered['destination_city'].unique()))

with col2:
    st.subheader("2. Airline & Time")
    if st.session_state.source_city and st.session_state.destination_city:
        filtered = df[
            (df['source_city'] == st.session_state.source_city) &
            (df['destination_city'] == st.session_state.destination_city)
        ]
        st.session_state.airline = st.selectbox("Airline", [""] + sorted(filtered['airline'].unique()))
        st.session_state.time_filter_type = st.radio("Filter by:", ("Departure", "Arrival"), horizontal=True)
        if st.session_state.time_filter_type == "Departure":
            st.session_state.departure_time = st.selectbox("Departure Time", [""] + sorted(df['departure_time'].unique()))
        else:
            st.session_state.arrival_time = st.selectbox("Arrival Time", [""] + sorted(df['arrival_time'].unique()))

with col3:
    st.subheader("3. Pricing Scenario")
    st.session_state.flight_class = st.selectbox("Class", [""] + sorted(df['class'].unique()))
    st.session_state.days_left = st.slider("Days Left Until Departure", 1, 50, st.session_state.days_left)

# --- Predict Button ---
if st.button("🔮 Predict & Optimize Price", type="primary"):
    if not all([st.session_state.source_city, st.session_state.destination_city, st.session_state.airline, st.session_state.flight_class]):
        st.warning("Please fill all dropdowns before predicting.")
    else:
        query = (
            (df['source_city'] == st.session_state.source_city) &
            (df['destination_city'] == st.session_state.destination_city) &
            (df['airline'] == st.session_state.airline)
        )
        if st.session_state.time_filter_type == "Departure" and st.session_state.departure_time:
            query &= (df['departure_time'] == st.session_state.departure_time)
        elif st.session_state.time_filter_type == "Arrival" and st.session_state.arrival_time:
            query &= (df['arrival_time'] == st.session_state.arrival_time)

        matched = df[query]

        if not matched.empty:
            record = matched.iloc[0]
            input_df = pd.DataFrame({
                'airline': [record['airline']], 'source_city': [record['source_city']],
                'departure_time': [record['departure_time']], 'stops': [record['stops']],
                'arrival_time': [record['arrival_time']], 'destination_city': [record['destination_city']],
                'class': [st.session_state.flight_class.lower()], 'duration': [record['duration']],
                'days_left': [st.session_state.days_left]
            })

            input_processed = preprocessor.transform(input_df)
            predicted_log_price = model.predict(input_processed)
            predicted_base_price = np.expm1(predicted_log_price)[0]
            optimization_result = find_optimal_price(predicted_base_price)

            st.session_state['prediction_results'] = {
                "base_price": predicted_base_price,
                "optimized_price": optimization_result['optimized_price'],
                "uplift": optimization_result['uplift_percent'],
                "input_df": input_df  # store for LLM later
            }
        else:
            st.error("No matching flight data found for the selected filters.")
            st.session_state['prediction_results'] = None

# --- Display Results ---
if st.session_state.get("prediction_results"):
    results = st.session_state["prediction_results"]
    st.subheader("Pricing Recommendation")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Base Price", f"₹{results['base_price']:,.0f}")
    col2.metric("✅ Optimized Price", f"₹{results['optimized_price']:,.0f}", delta=f"{results['uplift']:.2f}%")

    # --- LLM SHAP Explanation ---
    st.subheader("🧠 LLM Explanation")
    try:
        explainer = shap.TreeExplainer(model)
        # Fix SHAP error: convert to float before SHAP
        input_array = input_processed.astype(np.float32)
        shap_values = explainer(input_array)

        explanation_text = (
            "This prediction was influenced by features such as airline, departure time, "
            "arrival time, flight class, duration, and days left until departure. "
            "The model used SHAP to attribute importance to each feature, and the explanation "
            "below was generated using an LLM for clarity."
        )

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that explains SHAP model output in simple terms."},
                {"role": "user", "content": f"Explain this flight pricing prediction. {explanation_text}"}
            ],
            temperature=0.5
        )

        st.success(response.choices[0].message.content.strip())
    except Exception as e:
        st.error(f"LLM explanation could not be generated. Error:\n\n{e}")
