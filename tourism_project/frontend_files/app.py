
import streamlit as st
import requests

st.title("Tourism Package Prediction System")

# Input fields for product and store data
Sl_No = st.number_input("Sl No", min_value=0.0, value=12.66)
CustomerID	= st.number_input("Sl No", min_value=0.0, value=12.66)
#ProdTaken	= st.number_input("Sl No", min_value=0.0, value=12.66)
Age	= st.number_input("Sl No", min_value=0.0, value=12.66)
TypeofContact	= st.number_input("Sl No", min_value=0.0, value=12.66)
CityTier	= st.number_input("Sl No", min_value=0.0, value=12.66)
DurationOfPitch = st.number_input("Sl No", min_value=0.0, value=12.66)
Occupation	= st.number_input("Sl No", min_value=0.0, value=12.66)
Gender	= st.number_input("Sl No", min_value=0.0, value=12.66)
NumberOfPersonVisiting	= st.number_input("Sl No", min_value=0.0, value=12.66)
NumberOfFollowups	= st.number_input("Sl No", min_value=0.0, value=12.66)
ProductPitched	= st.number_input("Sl No", min_value=0.0, value=12.66)
PreferredPropertyStar	= st.number_input("Sl No", min_value=0.0, value=12.66)
MaritalStatus = st.number_input("Sl No", min_value=0.0, value=12.66)
NumberOfTrips	= st.number_input("Sl No", min_value=0.0, value=12.66)
Passport	= st.number_input("Sl No", min_value=0.0, value=12.66)
PitchSatisfactionScore	= st.number_input("Sl No", min_value=0.0, value=12.66)
OwnCar	= st.number_input("Sl No", min_value=0.0, value=12.66)
NumberOfChildrenVisiting	= st.number_input("Sl No", min_value=0.0, value=12.66)
Designation	= st.number_input("Sl No", min_value=0.0, value=12.66)
MonthlyIncome = st.number_input("Sl No", min_value=0.0, value=12.66)



Product_Test = st.number_input("Product Test", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027)
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Departmental Store","Food Mart"])
Product_Id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"])
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

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
    #TO OPTIMISE
    "Product_Test": Product_Test,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

if st.button("Predict", type='primary'):
    response = requests.post("https://huggingface.co/spaces/suman-komarla-adinarayana-groups/SumanKAGreatLearningInfo-EducationStudyAssignment10-TourismPackagePredictionAPI/v1/predict", json=product_data)  # Replace <user_name> and <space_name>
    if response.status_code == 200:
        result = response.json()
        predicted_sales = result["Sales"]
        st.write(f"Predicted Tourism Package Total: ₹{predicted_sales:.2f}")
    else:
        st.error("Error in API request")
