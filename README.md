## ✈️ Airline Pricing & Revenue Optimization Advisor

- A smart, data-driven pricing assistant that predicts base ticket fares and recommends optimized prices to maximize airline revenue — built using a Random Forest Regressor and deployed as an interactive Streamlit application.

---

## 📝 Project Overview

- In the dynamic airline industry, ticket pricing is a complex task influenced by demand, competition, timing, and service class. This project tackles this challenge by creating an end-to-end data science solution that empowers an airline's revenue management team.

       ✅ Accurate **price predictions**  
       ✅ Simulated **demand-based price optimization**  
       ✅ Transparent **SHAP explanations**  
       ✅ A user-friendly **Streamlit web app**

- The system leverages a machine learning model trained on historical flight data to provide an accurate Base Price Prediction. It then goes a step further by using an economic optimization engine to recommend a final price designed to maximize revenue. The entire solution is delivered through a user-friendly and interactive web application, turning complex data into actionable business insights.

---

## ✨ Key Features

## 🔮 Accurate Price Prediction: 

- Utilizes a highly accurate Random Forest Regressor (MAE: ₹1,185, R²: 0.985) to predict log-transformed base ticket prices.

## 💸 Revenue Optimization Engine: 

- Simulates demand based on price elasticity to propose an Optimized Price and calculates the estimated % Revenue Uplift, directly linking the model's output to business value.

## 📊 Explainable AI (XAI) with SHAP: 

- Integrates SHAP (SHapley Additive exPlanations) to provide a transparent breakdown of which features (e.g., class, airline, days_left) had the biggest impact on each individual price prediction, building user trust.

## 🌐 Cloud-Native Deployment: 

- The application is designed for the cloud, fetching the trained model and preprocessor files securely from a remote source (Azure Blob Storage) on startup.

## 🧠 Dynamic & Interactive UI: 

- The Streamlit interface features cascading filters that dynamically adjust the available flight options based on the user's route and airline selections, creating a realistic and intuitive workflow.

---

## ⚙️ Tech Stack

       | Category         | Tools Used                                      |
       |------------------|--------------------------------------------------|
       | Programming      | Python, Streamlit                               |
       | Modeling         | Random Forest Regressor                         |
       | Data Processing  | pandas, scikit-learn                            |
       | Explainability   | SHAP                                            |
       | Deployment       | Streamlit Cloud + Azure Blob Storage            |

---

## 🧠 Model & Evaluation

- After a comprehensive Exploratory Data Analysis (EDA) and testing multiple algorithms (Linear Regression, XGBoost, LightGBM), the Random Forest Regressor was selected as the champion model for its superior performance.

- **Target Variable:** Log-transformed ticket price (np.log1p) to handle the skewed price distribution.

- **Key Features:** airline, source_city, destination_city, stops, class, days_left, departure_time, arrival_time, and duration.

- **Final Evaluation Metrics (on the test set):**

- **Mean Absolute Error (MAE):** ₹1,185 (On average, the model's prediction is off by about ₹1,185).

- **R-squared (R²):** 0.985 (The model explains ~98.5% of the variance in ticket prices).

---

## 🏗️ System Architecture

       User Interface (Streamlit)
              │
              ▼
       1. User selects flight criteria (Route, Airline, Class, etc.)
              │
              ▼
       2. Input data is passed to the Preprocessing Pipeline (joblib)
              │ (One-Hot Encoding & Scaling)
              ▼
       3. Processed data is fed to the Random Forest Model (joblib) for Base Price Prediction
              │
              ├─► 4a. Base Price is sent to the Revenue Optimizer → Optimized Price is calculated
              └─► 4b. Processed data is sent to the SHAP Explainer → Feature contributions are calculated
              │
              ▼
       5. Results (Base Price, Optimized Price, and SHAP table) are displayed on the UI

---

## 🚀 Getting Started

- Follow these steps to run the application locally.

1. Clone the Repository

       git clone [https://github.com/your-username/airline-pricer.git](https://github.com/your-username/airline-pricer.git)
       cd airline-pricer

2. Create and Activate a Virtual Environment

# For Unix/macOS
       python3 -m venv venv
       source venv/bin/activate

# For Windows
       python -m venv venv
       venv\Scripts\activate

3. Install Dependencies

       pip install -r requirements.txt

4. Add Secret Keys

- Create a .streamlit/secrets.toml file in the root of your project folder. This is required for fetching the model from Azure Blob Storage when deploying.

       [azure]
       model_url = "https:...."
       preprocessor_url = "https:...."

5. Launch the App Locally

       streamlit run app.py

---

## 📦 Folder Structure

       airline-pricer/
       │
       ├── .streamlit/
       │   └── secrets.toml         # For storing API keys and URLs
       │
       ├── app.py                   # Main Streamlit application script
       │
       ├── data/
       │   └── Clean_Dataset_EDA_Processed.csv  # Cleaned data used by the app
       │
       ├── models/                  # (This folder is for local testing)
       │   ├── flight_price_model.joblib
       │   └── preprocessor.joblib
       │
       ├── notebooks/
       │   ├── Data-Exploration.ipynb
       │   └── Feature_Engineering_and_Modeling.ipynb
       │
       ├── .gitignore
       ├── requirements.txt
       └── README.md

## 📌 Future Work

- AI Advisor Integration: Implement a Generative AI (LLM) feature that takes the SHAP output as context and provides a full narrative explanation of the price prediction.

- Advanced SHAP Plots: Add more interactive SHAP visualizations like dependency plots and summary plots to a separate "Analysis" page in the app.

- A/B Testing
