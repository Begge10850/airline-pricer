import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import openai
import os
from dotenv import load_dotenv

# Load model and encoders
model = joblib.load("model/flight_price_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Airline Ticket Price Optimizer", layout="centered")
st.title("✈️ Airline Ticket Price Predictor & Optimizer")

st.markdown("""
This tool predicts the base price of a flight and then suggests an optimized price to maximize revenue using simulated economic logic.
""")

# User Inputs
st.subheader("🔍 Enter Flight Details")
col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("Airline", options=['IndiGo', 'Air India', 'SpiceJet', 'Vistara', 'GO FIRST'])
    source = st.selectbox("Source", options=['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Bangalore'])
    departure_time = st.selectbox("Departure Time", options=['Morning', 'Afternoon', 'Evening', 'Night', 'Early_Morning', 'Late_Night'])
    class_type = st.selectbox("Class", options=['Economy', 'Business'])

with col2:
    destination = st.selectbox("Destination", options=['Cochin', 'Delhi', 'New_Delhi', 'Hyderabad', 'Kolkata'])
    arrival_time = st.selectbox("Arrival Time", options=['Morning', 'Afternoon', 'Evening', 'Night', 'Early_Morning', 'Late_Night'])
    stops = st.selectbox("Stops", options=['zero', 'one', 'two_or_more'])
    duration = st.slider("Duration (Hours)", min_value=1.0, max_value=30.0, step=0.5)

days_left = st.slider("Days Left Until Flight", min_value=1, max_value=60)

# Encode user inputs for model
input_dict = {
    'airline': airline,
    'source_city': source,
    'departure_time': departure_time,
    'stops': stops,
    'arrival_time': arrival_time,
    'destination_city': destination,
    'class': class_type,
    'duration': duration,
    'days_left': days_left
}

encoded_input = []
input_df = pd.DataFrame([input_dict])

for col in ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']:
    le = label_encoders[col]
    encoded_input.append(le.transform([input_dict[col]])[0])
encoded_input.extend([duration, days_left])

# Optimizer function
def find_optimal_price(base_price, elasticity_factor=1.5):
    base_demand = 100
    max_revenue = base_price * base_demand
    best_price = base_price

    price_range = np.linspace(base_price * 0.8, base_price * 1.2, 100)
    revenues = []
    demands = []

    for price in price_range:
        price_diff_percent = (price - base_price) / base_price
        demand_factor = 1 - (price_diff_percent * elasticity_factor)
        demand = max(0, base_demand * demand_factor)
        revenue = price * demand
        revenues.append(revenue)
        demands.append(demand)
        if revenue > max_revenue:
            max_revenue = revenue
            best_price = price

    return round(best_price / 50) * 50, round(((max_revenue - base_price * base_demand) / (base_price * base_demand)) * 100, 2), price_range, revenues, demands

if st.button("🎯 Predict & Optimize"):
    base_price = model.predict([encoded_input])[0]
    optimized_price, uplift, price_range, revenues, demands = find_optimal_price(base_price)

    st.success("Prediction Complete")
    st.metric(label="Predicted Base Price", value=f"₹{round(base_price):,}")
    st.metric(label="Optimized Price for Maximum Revenue", value=f"₹{round(optimized_price):,}", delta=f"{uplift}%")

    # Revenue curve
    st.subheader("📈 Revenue & Demand Curve")
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(price_range, revenues, 'g-', label="Revenue")
    ax2.plot(price_range, demands, 'b--', label="Demand")

    ax1.set_xlabel('Price')
    ax1.set_ylabel('Revenue', color='g')
    ax2.set_ylabel('Demand', color='b')

    ax1.axvline(base_price, color='gray', linestyle=':', label="Base Price")
    ax1.axvline(optimized_price, color='red', linestyle='--', label="Optimized Price")
    fig.tight_layout()

    st.pyplot(fig)

    st.caption("🧠 Assumes price elasticity of 1.5 and base demand of 100 tickets at base price.")

    # SHAP + OpenAI Explanation
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer([encoded_input])

        top_features = pd.DataFrame({
            'feature': input_df.columns,
            'value': input_df.iloc[0].values,
            'shap_value': shap_values.values[0]
        }).sort_values(by='shap_value', key=abs, ascending=False).head(5)

        st.subheader("🔍 SHAP Feature Impact")
        for _, row in top_features.iterrows():
            st.write(f"• **{row.feature}** = {row.value} (impact: {row.shap_value:.2f})")

        explanation_text = "\n".join(
            [f"- {row.feature} = {row.value} (impact: {row.shap_value:.2f})" for _, row in top_features.iterrows()]
        )

        prompt = f"""A machine learning model predicted a flight price based on the following features and SHAP impacts:
{explanation_text}
Explain in simple business terms how these factors influence the ticket price.
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert data scientist who explains insights in simple airline industry terms."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        summary = response.choices[0].message.content
        st.subheader("🧠 LLM-Powered Explanation")
        st.markdown(summary)

    except Exception as e:
        st.warning("❗ SHAP or OpenAI explanation failed. Please check your API key or environment setup.")
