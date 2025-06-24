import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import requests

# --- Page Setup ---
st.set_page_config(page_title="Airline Pricing Advisor", layout="wide")
st.title("✈️ Dynamic Pricing & Revenue Advisor")
st.markdown("Predict base ticket prices and receive an optimized recommendation to maximize revenue.")

# --- Download Helpers ---
MODEL_PATH = "flight_price_model.joblib"
PREPROCESSOR_PATH = "preprocessor.joblib"
CSV_PATH = "data/Clean_Dataset_EDA_Processed.csv"  # Must exist in repo

def silent_download_from_azure(url, destination):
    if not os.path.exists(destination):
        response = requests.get(url)
        with open(destination, 'wb') as f:
            f.write(response.content)

# --- Load Artifacts ---
@st.cache_resource
def load_data_and_artifacts():
    try:
        model_url = st.secrets["azure"]["model_url"]
        preprocessor_url = st.secrets["azure"]["preprocessor_url"]
    except KeyError:
        st.error("⚠️ Azure model URLs are missing. Please set them in Streamlit Cloud secrets.")
        st.stop()

    silent_download_from_azure(model_url, MODEL_PATH)
    silent_download_from_azure(preprocessor_url, PREPROCESSOR_PATH)

    df = pd.read_csv(CSV_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return df, preprocessor, model

try:
    df, preprocessor, model = load_data_and_artifacts()
except FileNotFoundError:
    st.error("Required model or data files not found.")
    st.stop()

# --- Optimizer ---
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

# --- UI State Defaults ---
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

# --- Inputs ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Route")
    source_city_options = [""] + sorted(df['source_city'].unique())
    st.session_state.source_city = st.selectbox("Source City", source_city_options)
    if st.session_state.source_city:
        destination_options = [""] + sorted(
            df[df['source_city'] == st.session_state.source_city]['destination_city'].unique()
        )
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

# --- Prediction ---
if st.button("🔮 Predict & Optimize Price", type="primary"):
    if not all([
        st.session_state.source_city,
        st.session_state.destination_city,
        st.session_state.airline,
        st.session_state.flight_class
    ]):
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

            input_processed_raw = preprocessor.transform(input_df)
            input_processed = input_processed_raw.toarray() if hasattr(input_processed_raw, 'toarray') else input_processed_raw
            input_processed = input_processed.astype(np.float32)

            predicted_log_price = model.predict(input_processed)
            predicted_base_price = np.expm1(predicted_log_price)[0]

            optimization_result = find_optimal_price(predicted_base_price)

            st.session_state['prediction_results'] = {
                "base_price": predicted_base_price,
                "optimized_price": optimization_result['optimized_price'],
                "uplift": optimization_result['uplift_percent'],
                "input_df": input_df,
                "input_processed": input_processed
            }
        else:
            st.error("No matching flight data found for the selected filters.")
            st.session_state['prediction_results'] = None

# --- Results ---
if st.session_state.get('prediction_results'):
    results = st.session_state['prediction_results']
    st.subheader("Pricing Recommendation")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Predicted Base Price", value=f"₹{results['base_price']:,.0f}")
    with res_col2:
        st.metric(label="✅ Optimized Price", value=f"₹{results['optimized_price']:,.0f}", delta=f"{results['uplift']:.2f}%")

    st.subheader("🧮 SHAP Feature Contributions")
    try:
        input_processed = results['input_processed'].astype(np.float32)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_processed)

        encoded_feature_names = preprocessor.get_feature_names_out()
        original_feature_names = [name.split("_", 1)[-1] if "_" in name else name for name in encoded_feature_names]
        grouped = {}

        for feature, original_name, value in zip(encoded_feature_names, original_feature_names, contrib_raw):
            grouped.setdefault(original_name, 0)
            grouped[original_name] += value

        contrib_df = pd.DataFrame(list(grouped.items()), columns=["Original Feature", "Contribution (₹)"])
        contrib_df["Contribution (₹)"] = contrib_df["Contribution (₹)"].round(2)
        contrib_df = contrib_df.sort_values("Contribution (₹)", ascending=False)

        st.dataframe(contrib_df)
        st.caption("Sum of contributions explains the predicted ticket price. Feature contributions are grouped by original input fields.")
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
