import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Setup ---
st.set_page_config(page_title="Airline Pricing Advisor", layout="wide")
st.title("✈️ Dynamic Pricing & Revenue Advisor")
st.markdown("Predict base prices and receive an optimized recommendation to maximize revenue.")

# --- Load Data and Model ---
@st.cache_resource
def load_data_and_artifacts():
    # Using the final clean dataset name you provided in your app
    df = pd.read_csv("data/Clean_Dataset_EDA_Processed.csv") 
    preprocessor = joblib.load("preprocessor.joblib")
    model = joblib.load("flight_price_model.joblib")
    return df, preprocessor, model

try:
    df, preprocessor, model = load_data_and_artifacts()
except FileNotFoundError:
    st.error("Required model or data files not found.")
    st.stop()


# --- THE NEW OPTIMIZER FUNCTION ---
def find_optimal_price(base_price, elasticity_factor=1.5, price_range_pct=0.25):
    """
    Simulates demand across a price range to find the revenue-maximizing price.
    
    Args:
        base_price (float): The price predicted by our ML model.
        elasticity_factor (float): How much demand drops for a 1% price increase.
        price_range_pct (float): The range to search around the base price (e.g., 0.25 for +/- 25%).

    Returns:
        dict: A dictionary with the optimal price and projected revenue uplift.
    """
    best_price = base_price
    max_revenue = 0
    
    # Assume a baseline demand (e.g., 100 theoretical units) at the base price
    base_demand = 100 
    base_revenue = base_price * base_demand
    
    # Test prices in a range around the base price
    price_range = np.linspace(base_price * (1 - price_range_pct), base_price * (1 + price_range_pct), 100)
    
    for price in price_range:
        # Calculate the change in demand based on the price difference
        price_diff_percent = (price - base_price) / base_price
        demand_factor = 1 - (price_diff_percent * elasticity_factor)
        
        # Ensure demand doesn't become negative
        demand_at_price = max(0, base_demand * demand_factor)
        expected_revenue = price * demand_at_price
        
        if expected_revenue > max_revenue:
            max_revenue = expected_revenue
            best_price = price
            
    uplift = ((max_revenue - base_revenue) / base_revenue) * 100 if base_revenue > 0 else 0
    
    return {"optimized_price": best_price, "uplift_percent": uplift}


# --- UI and Logic (Based on your excellent app.py) ---

# --- Initialize Session State ---
defaults = {
    "source_city": "", "destination_city": "", "airline": "",
    "time_filter_type": "Departure", "departure_time": "", "arrival_time": "",
    "flight_class": "Economy", "days_left": 15, "submitted": False, 
    "prediction_results": None # New state to hold all results
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.button("🔄 Reset Selections"):
    for key, value in defaults.items():
        st.session_state[key] = value
        st.experimental_rerun() # Rerun to clear widgets

# --- Input Panels ---
st.subheader("Fill the details below and press 'Predict Price'")
col1, col2, col3 = st.columns(3)
# ... (Your UI code for cols 1-7 and the time filter goes here. It is correct as-is.)
# ...

# --- For this example, let's assume all the session_state variables are filled by your UI
# --- and we are at the prediction step. ---

# --- Prediction & Optimization Logic ---
if st.button("🔮 Predict & Optimize Price", type="primary"):
    # (Your validation logic to check if all fields are filled)
    
    # Find the matching flight record based on user's filters
    query = (
        (df['source_city'] == st.session_state.source_city) &
        (df['destination_city'] == st.session_state.destination_city) &
        (df['airline'] == st.session_state.airline)
    )
    # Add time filters etc. from your app's logic
    matched = df[query]

    if not matched.empty:
        record = matched.iloc[0] # Use a matched record
        
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
        
        # === NEW STEP: Call the optimizer function ===
        optimization_result = find_optimal_price(predicted_base_price)
        
        # Store all results in session state to display them
        st.session_state['prediction_results'] = {
            "base_price": predicted_base_price,
            "optimized_price": optimization_result['optimized_price'],
            "uplift": optimization_result['uplift_percent']
        }
    else:
        st.error("No matching flight data found for the selected filters.")
        st.session_state['prediction_results'] = None


# --- Display Results ---
if st.session_state.get('prediction_results'):
    results = st.session_state['prediction_results']
    st.subheader("Pricing Recommendation")
    
    # Use columns for a clean side-by-side display
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="Predicted Base Price", value=f"₹{results['base_price']:,.0f}")
    with col_res2:
        st.metric(label="✅ Optimized Price Recommendation", 
                  value=f"₹{results['optimized_price']:,.0f}", 
                  delta=f"{results['uplift']:.2f}% Revenue Uplift")

