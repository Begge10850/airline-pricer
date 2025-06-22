import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# Load model and preprocessor
model = joblib.load("flight_price_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")

st.set_page_config(page_title="Dynamic Pricing Advisor", layout="wide")

st.title("✈️ Dynamic Pricing & Revenue Advisor")
st.caption("Predict base ticket prices and receive an optimized recommendation to maximize revenue.")

# --- Form UI ---
with st.form(key="prediction_form"):
    st.subheader("1. Route")
    source = st.selectbox("Source City", ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])
    destination = st.selectbox("Destination City", ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai', 'Hyderabad'])

    st.subheader("2. Airline & Time")
    airline = st.selectbox("Airline", ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Vistara', 'GO_FIRST', 'Multiple carriers'])
    dep_arr = st.radio("Filter by:", ["Departure", "Arrival"])
    dep_time = st.selectbox(f"{dep_arr} Time", ['Morning', 'Evening', 'Afternoon', 'Night'])

    st.subheader("3. Pricing Scenario")
    ticket_class = st.selectbox("Class", ['economy', 'business'])
    days_left = st.slider("Days Left Until Departure", 1, 50, 15)

    submitted = st.form_submit_button("🎯 Predict & Optimize Price")

# --- Prediction ---
if submitted:
    # Map to match training encodings
    airline = airline.upper().replace(" ", "_")
    dep_arr_time = dep_time
    if dep_arr == "Arrival":
        dep_arr_time = f"Arrival_{dep_time}"
    else:
        dep_arr_time = f"Dep_{dep_time}"

    input_data = {
        "airline": airline,
        "source_city": source,
        "destination_city": destination,
        "departure_time": dep_time,
        "arrival_time": dep_time,
        "class": ticket_class,
        "days_left": days_left,
        "duration": 2.5,  # Placeholder
        "total_stops": 1  # Placeholder
    }

    df = pd.DataFrame([input_data])

    # Preprocess
    input_processed = preprocessor.transform(df)
    prediction = model.predict(input_processed)[0]
    optimized_price = prediction * 0.83  # 17% reduction

    col1, col2 = st.columns(2)
    col1.metric("💰 Predicted Base Price", f"₹{int(prediction):,}")
    col2.metric("✅ Optimized Price", f"₹{int(optimized_price):,}", f"{-17}%")

    # --- SHAP ---
    st.markdown("### 🧱 SHAP Feature Contributions")
    try:
        explainer = shap.Explainer(model)
        input_array = input_processed.astype(np.float32)  # Fix dtype casting
        shap_values = explainer(input_array)
        contributions = shap_values.values[0]
        feature_names = preprocessor.get_feature_names_out()

        df_shap = pd.DataFrame({
            "Feature": feature_names,
            "Contribution (₹)": contributions.round(2)
        }).sort_values("Contribution (₹)", key=lambda x: abs(x), ascending=False)

        total_contribution = df_shap["Contribution (₹)"].sum()
        st.write(f"Sum of contributions: ₹{total_contribution:.2f}")
        st.dataframe(df_shap, use_container_width=True)

    except Exception as e:
        st.error(f"SHAP explanation failed: {str(e)}")

# --- Reset button ---
if st.button("🔄 Reset Form"):
    st.experimental_rerun()
