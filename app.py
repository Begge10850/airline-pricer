import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import openai
import os
from dotenv import load_dotenv

# Load OpenAI API Key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Page Setup ---
st.set_page_config(page_title="Airline Pricing Advisor", layout="wide")
st.title("✈️ Dynamic Pricing & Revenue Advisor")
st.markdown("Predict base ticket prices and receive an optimized recommendation to maximize revenue.")

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
    st.error("Required model or data files not found.")
    st.stop()

# --- Optimizer Function ---
def find_optimal_price(base_price, elasticity_factor=1.5, price_range_pct=0.25):
    base_demand = 100 
    base_revenue = base_price * base_demand
    best_price = base_price
    max_revenue = 0

    price_range = np.linspace(base_price * (1 - price_range_pct), base_price * (1 + price_range_pct), 100)
    for price in price_range:
        price_diff_percent = (price - base_price) / base_price
        demand_factor = 1 - (price_diff_percent * elasticity_factor)
        demand = max(0, base_demand * demand_factor)
        revenue = price * demand
        if revenue > max_revenue:
            max_revenue = revenue
            best_price = price

    uplift = ((max_revenue - base_revenue) / base_revenue) * 100 if base_revenue > 0 else 0
    return {"optimized_price": best_price, "uplift_percent": uplift}

# --- LLM Explanation Function ---
def explain_with_llm(top_features):
    prompt = f"""You are an AI assistant helping a pricing analyst understand machine learning decisions.
The model predicted a high or low flight price because of these important features: {top_features}.
Explain in simple terms how these features could influence flight prices. Keep the explanation concise and insightful."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for airline pricing analysts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"LLM explanation could not be generated. Error:\n\n{e}"

# --- Streamlit UI ---
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

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1. Route")
    source_city_options = [""] + sorted(df['source_city'].unique())
    st.session_state.source_city = st.selectbox("Source City", source_city_options)
    if st.session_state.source_city:
        destination_options = [""] + sorted(df[df['source_city'] == st.session_state.source_city]['destination_city'].unique())
        st.session_state.destination_city = st.selectbox("Destination City", destination_options)

with col2:
    st.subheader("2. Airline & Time")
    if st.session_state.source_city and st.session_state.destination_city:
        airline_options = [""] + sorted(df[
            (df['source_city'] == st.session_state.source_city) &
            (df['destination_city'] == st.session_state.destination_city)
        ]['airline'].unique())
        st.session_state.airline = st.selectbox("Airline", airline_options)
        st.session_state.time_filter_type = st.radio("Filter by:", ("Departure", "Arrival"), horizontal=True)
        if st.session_state.time_filter_type == "Departure":
            dep_options = [""] + sorted(df['departure_time'].unique())
            st.session_state.departure_time = st.selectbox("Departure Time", dep_options)
        else:
            arr_options = [""] + sorted(df['arrival_time'].unique())
            st.session_state.arrival_time = st.selectbox("Arrival Time", arr_options)

with col3:
    st.subheader("3. Pricing Scenario")
    class_options = [""] + sorted(df['class'].unique())
    st.session_state.flight_class = st.selectbox("Class", class_options)
    st.session_state.days_left = st.slider("Days Left Until Departure", 1, 50, st.session_state.days_left)

# --- Predict & Explain ---
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
                'class': [st.session_state.flight_class.lower()],
                'duration': [record['duration']],
                'days_left': [st.session_state.days_left]
            })

            input_processed = preprocessor.transform(input_df)
            if hasattr(input_processed, 'toarray'):
                input_processed = input_processed.toarray()
            input_processed = np.array(input_processed).reshape(1, -1)

            predicted_log_price = model.predict(input_processed)[0]
            predicted_base_price = np.expm1(predicted_log_price)

            optimization_result = find_optimal_price(predicted_base_price)

            # SHAP + LLM
            explainer = shap.Explainer(model, feature_names=preprocessor.get_feature_names_out())
            shap_values = explainer(input_processed)
            top_indices = np.argsort(np.abs(shap_values.values[0]))[::-1][:5]
            top_features = [explainer.feature_names[i] for i in top_indices]

            llm_explanation = explain_with_llm(top_features)

            # Display
            st.session_state['prediction_results'] = {
                "base_price": predicted_base_price,
                "optimized_price": optimization_result['optimized_price'],
                "uplift": optimization_result['uplift_percent'],
                "llm_explanation": llm_explanation
            }
        else:
            st.error("No matching flight data found for the selected filters.")
            st.session_state['prediction_results'] = None

# --- Display ---
if st.session_state.get('prediction_results'):
    res = st.session_state['prediction_results']
    st.subheader("💰 Pricing Recommendation")
    col1, col2 = st.columns(2)
    col1.metric("Base Price", f"₹{res['base_price']:,.0f}")
    col2.metric("Optimized Price", f"₹{res['optimized_price']:,.0f}", delta=f"{res['uplift']:.2f}%")

    st.subheader("🧠 LLM Explanation")
    st.markdown(res["llm_explanation"])
