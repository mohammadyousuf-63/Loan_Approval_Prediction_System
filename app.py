# Stage 8: Streamlit Interface

# Import required libraries
import streamlit as st
import pandas as pd
import joblib

# Stage 8: Load the trained model and preprocessing objects

random_forest_model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoding_mappings = joblib.load("models/encoding_mappings.pkl")

# Stage 8: Set the application title

st.title("Loan Approval Prediction System")
st.write("Enter the applicant details below to predict loan approval.")

# Stage 8: Collect applicant details

st.header("Applicant Details")

applicant_income = st.number_input("Applicant Income", min_value=0.0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)

employment_status = st.selectbox(
    "Employment Status",
    list(encoding_mappings["Employment_Status"].keys())
)

age = st.number_input("Age", min_value=18, max_value=100, value=30)

marital_status = st.selectbox(
    "Marital Status",
    list(encoding_mappings["Marital_Status"].keys())
)

dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)

credit_score = st.number_input("Credit Score", min_value=0.0, max_value=1000.0)

existing_loans = st.number_input("Existing Loans", min_value=0, value=0)

dti_ratio = st.number_input("DTI Ratio", min_value=0.0)

savings = st.number_input("Savings", min_value=0.0)

collateral_value = st.number_input("Collateral Value", min_value=0.0)

loan_amount = st.number_input("Loan Amount", min_value=0.0)

loan_term = st.number_input("Loan Term", min_value=1, value=12)

loan_purpose = st.selectbox(
    "Loan Purpose",
    list(encoding_mappings["Loan_Purpose"].keys())
)

property_area = st.selectbox(
    "Property Area",
    list(encoding_mappings["Property_Area"].keys())
)

education_level = st.selectbox(
    "Education Level",
    list(encoding_mappings["Education_Level"].keys())
)

gender = st.selectbox(
    "Gender",
    list(encoding_mappings["Gender"].keys())
)

employer_category = st.selectbox(
    "Employer Category",
    list(encoding_mappings["Employer_Category"].keys())
)

# Stage 8: Add the prediction button

if st.button("Predict Loan Approval"):

    st.write("Processing application...")

    # Stage 8: Convert categorical inputs into encoded values

    employment_status_encoded = encoding_mappings["Employment_Status"][employment_status]
    marital_status_encoded = encoding_mappings["Marital_Status"][marital_status]
    loan_purpose_encoded = encoding_mappings["Loan_Purpose"][loan_purpose]
    property_area_encoded = encoding_mappings["Property_Area"][property_area]
    education_level_encoded = encoding_mappings["Education_Level"][education_level]
    gender_encoded = encoding_mappings["Gender"][gender]
    employer_category_encoded = encoding_mappings["Employer_Category"][employer_category]

    # Stage 8: Create the input DataFrame
    
    
    input_data = pd.DataFrame([{
         "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment_status_encoded,
        "Age": age,
        "Marital_Status": marital_status_encoded,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose_encoded,
        "Property_Area": property_area_encoded,
        "Education_Level": education_level_encoded,
        "Gender": gender_encoded,
        "Employer_Category": employer_category_encoded
    }])

        # Stage 8: Scale the input data

    input_scaled = scaler.transform(input_data)
        # Stage 8: Make the loan approval prediction

    prediction = random_forest_model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
# Stage 9: Load the dataset for dashboard statistics

dashboard_df = pd.read_csv("data/loan_approval_data.csv")

dashboard_df = dashboard_df.dropna(subset=["Loan_Approved"])

# Stage 9: Display approval statistics

st.header("Dashboard")

total_applicants = len(dashboard_df)

approved_applicants = (
    dashboard_df["Loan_Approved"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("yes")
    .sum()
)

rejected_applicants = (
    dashboard_df["Loan_Approved"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["0", "no"])
    .sum()
)

approval_rate = (approved_applicants / total_applicants) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Applicants", total_applicants)
col2.metric("Approved", approved_applicants)
col3.metric("Rejected", rejected_applicants)
col4.metric("Approval Rate", f"{approval_rate:.2f}%")

# Stage 9: Display approval statistics by gender

st.subheader("Approval Statistics by Gender")

gender_approval = (
    dashboard_df.groupby("Gender")["Loan_Approved"]
    .apply(lambda x: (x.astype(str).str.strip().str.lower() == "yes").mean() * 100)
)

st.bar_chart(gender_approval)

# Stage 9: Display approval statistics by education level

st.subheader("Approval Statistics by Education Level")

education_approval = (
    dashboard_df.groupby("Education_Level")["Loan_Approved"]
    .apply(lambda x: (x.astype(str).str.strip().str.lower() == "yes").mean() * 100)
)

st.bar_chart(education_approval)
# Stage 9: Display approval statistics by employment status

st.subheader("Approval Statistics by Employment Status")

employment_approval = (
    dashboard_df.groupby("Employment_Status")["Loan_Approved"]
    .apply(lambda x: (x.astype(str).str.strip().str.lower() == "yes").mean() * 100)
)

st.bar_chart(employment_approval)
# Stage 9: Display approval statistics by property area

st.subheader("Approval Statistics by Property Area")

property_approval = (
    dashboard_df.groupby("Property_Area")["Loan_Approved"]
    .apply(lambda x: (x.astype(str).str.strip().str.lower() == "yes").mean() * 100)
)

st.bar_chart(property_approval)
# Stage 9: Display approval trends by age group

st.subheader("Approval Trends by Age Group")

dashboard_df["Age_Group"] = pd.cut(
    dashboard_df["Age"],
    bins=[0, 25, 35, 45, 55, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56+"]
)

age_approval = (
    dashboard_df.groupby("Age_Group", observed=False)["Loan_Approved"]
    .apply(lambda x: (x.astype(str).str.strip().str.lower() == "yes").mean() * 100)
)

st.bar_chart(age_approval)
# Stage 10: Display important factors affecting loan approval

st.subheader("Factors Affecting Loan Approval")

feature_names = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Employment_Status",
    "Age",
    "Marital_Status",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category"
]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": random_forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(feature_importance)

st.bar_chart(
    feature_importance.set_index("Feature")
)

# Stage 10: Display risk analysis

st.subheader("Risk Analysis")

dashboard_df["DTI_Risk_Group"] = pd.cut(
    dashboard_df["DTI_Ratio"],
    bins=[-float("inf"), 0.30, 0.50, float("inf")],
    labels=["Low DTI", "Medium DTI", "High DTI"]
)

dti_approval = (
    dashboard_df.groupby("DTI_Risk_Group", observed=False)["Loan_Approved"]
    .apply(lambda x: (x.astype(str).str.strip().str.lower() == "yes").mean() * 100)
)

st.bar_chart(dti_approval)

st.write(
    "The chart shows loan approval rates across different DTI ratio groups. "
    "DTI ratio is used to understand the relationship between existing debt "
    "obligations and loan approval outcomes in this dataset."
)

# Stage 10: Display business recommendations

st.subheader("Business Recommendations")

st.write(
    "• Consider credit score and debt-to-income ratio when reviewing loan applications."
)

st.write(
    "• Review applicant income, existing loans, and loan amount together to understand financial capacity."
)

st.write(
    "• Use applicant demographics and employment information along with financial factors for application analysis."
)

st.write(
    "• Use the machine learning prediction as a decision-support tool and review applications using appropriate financial criteria."
)

st.write(
    "• Monitor approval patterns across different applicant groups to identify changes in loan approval outcomes."
)