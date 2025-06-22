import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import openai
import os
from dotenv import load_dotenv

# Load OpenAI key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Airline Pricing Advisor", layout="wide")
st.title("✈️ Dynamic Pricing & Revenue Advisor")

# Load data, model, preprocessor
@st.cache_resource
def load_assets():
    df = pd.read_csv("data/Clean_Dataset_EDA_Processed.csv")
    preprocessor = joblib.load("preprocessor.joblib")
    model = joblib.load("flight_price_model.joblib")
    return df, preprocessor, model

df, preprocessor, model = load_assets()

# Optimizer
def find_optimal_price(base_price, elasticity=1.5):
    base_demand = 100
    price_range = np.linspace(base_price * 0.8, base_price * 1.2, 100)
    best_price, max_revenue = base_price, 0

    for price in price_range:
        demand_factor = 1 - ((price - base_price) / base_price) * elasticity
        demand = max(0, base_demand * demand_factor)
        revenue = price * demand
        if revenue > max_revenue:
            max_revenue = revenue
            best_price = price

    uplift = ((max_revenue - base_price * base_demand) / (base_price * base_demand)) * 100
    return round(best_price), round(uplift, 2)

# UI
st.markdown("#### Fill in the flight details below:")
col1, col2, col3 = st.columns(3)

with col1:
    source = st.selectbox("Source City", sorted(df['source_city'].unique()))
    destination = st.selectbox("Destination City", sorted(df[df['source_city'] == source]['destination_city'].unique()))
with col2:
    airline = st.selectbox("Airline", sorted(df['airline'].unique()))
    departure_time = st.selectbox("Departure Time", sorted(df['departure_time'].unique()))
    arrival_time = st.selectbox("Arrival Time", sorted(df['arrival_time'].unique()))
with col3:
    cls = st.selectbox("Class", sorted(df['class'].unique()))
    stops = st.selectbox("Stops", sorted(df['stops'].unique()))
    duration = st.slider("Duration (hours)", 1.0, 30.0, step=0.5)
    days_left = st.slider("Days Left", 1, 60)

# Run Prediction
if st.button("🔮 Predict & Explain"):
    input_df = pd.DataFrame({
        'airline': [airline], 'source_city': [source],
        'departure_time': [departure_time], 'stops': [stops],
        'arrival_time': [arrival_time], 'destination_city': [destination],
        'class': [cls], 'duration': [duration], 'days_left': [days_left]
    })

    try:
        input_transformed = preprocessor.transform(input_df)
        log_price = model.predict(input_transformed)[0]
        base_price = np.expm1(log_price)
        optimized_price, uplift = find_optimal_price(base_price)

        st.metric("Base Price", f"₹{round(base_price)}")
        st.metric("Optimized Price", f"₹{optimized_price}", delta=f"{uplift}%")

        # SHAP
        explainer = shap.Explainer(model)
        shap_vals = explainer(input_transformed)
        base_log = shap_vals.base_values[0]
        shap_contribs = shap_vals.values[0]

        # Map back to user input
        transformed_cols = shap_vals.feature_names
        original_cols = input_df.columns.tolist()
        df_sparse = pd.DataFrame(input_transformed.toarray(), columns=transformed_cols)

        # Group by original columns
        contrib_dict = {}
        for feat in original_cols:
            related = [col for col in df_sparse.columns if col.startswith(f"cat_{feat}_") or col == feat]
            log_contrib = sum([shap_contribs[df_sparse.columns.get_loc(col)] for col in related])
            price_contrib = np.expm1(base_log + log_contrib) - np.expm1(base_log)
            contrib_dict[feat] = price_contrib

        # LLM Explanation Prompt
        items = [f"{k} contributes ₹{v:,.0f}" for k, v in contrib_dict.items()]
        user_friendly_input = "\n".join(items)
        prompt = f"""
You are a pricing analyst AI. Based on SHAP values, explain to a business user how each input affected the ticket price prediction.
Here is the breakdown:
{user_friendly_input}

Generate a human-friendly explanation.
"""

        with st.spinner("Generating AI Explanation..."):
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "system", "content": "You are a helpful pricing analyst AI."},
                              {"role": "user", "content": prompt}]
                )
                explanation = completion.choices[0].message.content.strip()
                st.subheader("🧠 LLM Explanation")
                st.write(explanation)
            except Exception as e:
                st.warning(f"LLM explanation could not be generated. Error:\n\n{str(e)}")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
