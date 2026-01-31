import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="suman-komarla-adinarayana-groups/SumanKAGreatLearningInfo-EducationStudyAssignment10-TourismPackagePredictionAPI", filename="tourism_project_huggingface_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Machine Failure Prediction App")
st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# Input fields for product and store data
Sl_No = st.number_input("Sl_No", min_value=0.0, value=1000000.00)
CustomerID	= st.number_input("CustomerID", min_value=0.0, value=1000000.00)
#ProdTaken	= st.number_input("ProdTaken", min_value=0.0, value=1000000.00)
Age	= st.number_input("Age", min_value=0.0, value=1000000.00)
TypeofContact	= st.number_input("TypeofContact", min_value=0.0, value=1000000.00)
CityTier	= st.number_input("CityTier", min_value=0.0, value=1000000.00)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=0.0, value=1000000.00)
Occupation	= st.number_input("Occupation", min_value=0.0, value=1000000.00)
Gender	= st.number_input("Gender", min_value=0.0, value=1000000.00)
NumberOfPersonVisiting	= st.number_input("NumberOfPersonVisiting", min_value=0.0, value=1000000.00)
NumberOfFollowups	= st.number_input("NumberOfFollowups", min_value=0.0, value=1000000.00)
ProductPitched	= st.number_input("ProductPitched", min_value=0.0, value=1000000.00)
PreferredPropertyStar	= st.number_input("PreferredPropertyStar", min_value=0.0, value=1000000.00)
MaritalStatus = st.number_input("MaritalStatus", min_value=0.0, value=1000000.00)
NumberOfTrips	= st.number_input("NumberOfTrips", min_value=0.0, value=1000000.00)
Passport	= st.number_input("Passport", min_value=0.0, value=1000000.00)
PitchSatisfactionScore	= st.number_input("PitchSatisfactionScore", min_value=0.0, value=1000000.00)
OwnCar	= st.number_input("OwnCar", min_value=0.0, value=1000000.00)
NumberOfChildrenVisiting	= st.number_input("NumberOfChildrenVisiting", min_value=0.0, value=1000000.00)
Designation	= st.number_input("Designation", min_value=0.0, value=1000000.00)
MonthlyIncome = st.number_input("MonthlyIncome", min_value=0.0, value=1000000.00)



product_data = {
    "Sl_No": Sl_No,
    "CustomerID" : CustomerID,
    "Age" : Age,
    "TypeofContact" : TypeofContact,
    "CityTier" : CityTier,
    "DurationOfPitch" : DurationOfPitch,
    "Occupation" : Occupation,
    "Gender" : Gender,
    "NumberOfPersonVisiting" : NumberOfPersonVisiting,
    "NumberOfFollowups" : NumberOfFollowups,
    "ProductPitched" : ProductPitched,
    "PreferredPropertyStar" : PreferredPropertyStar,
    "MaritalStatus" : MaritalStatus,
    "NumberOfTrips" : NumberOfTrips,
    "Passport" : Passport,
    "PitchSatisfactionScore" : PitchSatisfactionScore,
    "OwnCar" : OwnCar,
    "NumberOfChildrenVisiting" : NumberOfChildrenVisiting,
    "Designation" : Designation,
    "MonthlyIncome" : MonthlyIncome,
}





if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "ProdTaken" if prediction == 1 else "No ProdTaken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
