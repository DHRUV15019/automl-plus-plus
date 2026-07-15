import streamlit as st
import requests

# Page settings
st.set_page_config(page_title="AutoML++ Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details below to see if they would survive, powered by my **live AutoML++ API** hosted on Render.")
st.divider()

# User Inputs using Streamlit widgets
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3], help="1 = 1st, 2 = 2nd, 3 = 3rd")
    sex = st.selectbox("Sex", ["male", "female"])

with col2:
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=25.0, step=1.0)
    fare = st.number_input("Passenger Fare ($)", min_value=0.0, value=30.0, step=1.0)

st.divider()

# Submit Button
if st.button("Predict Survival 🔮", use_container_width=True):
    
    # 1. Prepare the exact data payload expected by your FastAPI backend
    # API exactly "data" key expect kar rahi hai, toh usko wahi de
    payload = {
        "data": [
            {
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "Fare": fare
            }
        ]
    }
    
    # 2. Your LIVE Render API URL
    api_url = "https://automl-plus-plus.onrender.com/predict"
    
    # 3. Make the API call and handle the response
    try:
        with st.spinner("Connecting to live cloud model..."):
            response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extracting the prediction from the JSON response
            prediction = result.get("prediction", None)
            
            if prediction == 1:
                st.success("🎉 **Prediction: SURVIVED!** This passenger would likely survive.")
            elif prediction == 0:
                st.error("💀 **Prediction: DID NOT SURVIVE.** This passenger would likely not make it.")
            else:
                st.info(f"API Response: {result}")
        else:
            st.warning(f"Error from API (Status {response.status_code}): {response.text}")
            
    except Exception as e:
        st.error(f"Failed to connect to backend. Error: {e}")