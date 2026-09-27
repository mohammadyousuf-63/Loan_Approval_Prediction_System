# Loan Approval Prediction System

A machine learning system that predicts loan approval using applicant financial and personal information, with a Streamlit dashboard for predictions, analysis, and insights.

## Live Demo

**Streamlit App:** https://loanapprovalpredictionsystem-p2cmt9zzm6jftoapykswbx.streamlit.app/

## GitHub Repository

**GitHub:** https://github.com/mohammadyousuf-63/Loan_Approval_Prediction_System

## Project Overview

The Loan Approval Prediction System is a machine learning project that analyzes applicant information and predicts whether a loan application is likely to be approved or rejected.

The project includes data preprocessing, exploratory data analysis, classification model training, model evaluation, applicant prediction, and a Streamlit dashboard for displaying loan approval statistics and insights.

## Project Objectives

* Analyze loan applicant data
* Handle missing values and prepare data for machine learning
* Perform exploratory data analysis
* Identify patterns in loan approval outcomes
* Build classification models for loan approval prediction
* Compare model performance using evaluation metrics
* Provide an interactive prediction interface
* Display loan approval statistics and analytical insights through a dashboard

## Features

### 1. Data Preprocessing

The dataset is prepared for machine learning by:

* Handling missing values
* Removing records with missing loan approval status
* Filling missing numerical values using median values
* Filling missing categorical values using mode values
* Encoding categorical features
* Separating input features and target variable
* Scaling numerical and encoded features

### 2. Exploratory Data Analysis

The project analyzes:

* Applicant income distribution
* Loan amount distribution
* Credit score distribution
* Overall loan approval rate
* Approval rates by employment status
* Approval rates by property area
* Approval rates by education level
* Approval rates by gender
* Applicant age and loan approval
* Correlation between numerical features

Visualizations are created using Matplotlib and Seaborn.

### 3. Machine Learning Models

The following classification models were implemented:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)

### 4. Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The evaluation results are compared to understand the prediction performance of each model.

### 5. Loan Approval Prediction

The Streamlit application allows users to enter applicant information such as:

* Applicant Income
* Coapplicant Income
* Employment Status
* Age
* Marital Status
* Dependents
* Credit Score
* Existing Loans
* DTI Ratio
* Savings
* Collateral Value
* Loan Amount
* Loan Term
* Loan Purpose
* Property Area
* Education Level
* Gender
* Employer Category

The trained Random Forest model processes the entered information and displays the prediction as:

* **Loan Approved**
* **Loan Rejected**

### 6. Dashboard

The Streamlit dashboard displays:

* Total number of applicants
* Number of approved applications
* Number of rejected applications
* Overall approval rate
* Approval statistics by gender
* Approval statistics by education level
* Approval statistics by employment status
* Approval statistics by property area
* Approval trends by age group
* Important factors affecting loan approval
* Risk analysis based on DTI ratio
* Business recommendations

## Dataset

The project uses a loan approval dataset containing **1,000 records and 20 columns**.

The dataset includes:

* Applicant ID
* Applicant Income
* Coapplicant Income
* Employment Status
* Age
* Marital Status
* Dependents
* Credit Score
* Existing Loans
* DTI Ratio
* Savings
* Collateral Value
* Loan Amount
* Loan Term
* Loan Purpose
* Property Area
* Education Level
* Gender
* Employer Category
* Loan Approved

After preprocessing and removing records with missing loan approval status, **950 records** were used for the machine learning workflow.

## Dataset Analysis

The overall loan approval rate in the processed dataset is approximately **31.37%**.

The exploratory analysis examines how approval outcomes vary across applicant characteristics and financial factors.

The correlation analysis also helps identify relationships between numerical features and loan approval outcomes.

## Model Performance

The implemented models were evaluated on the test dataset.

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   87.37% |    84.62% | 73.33% |   78.57% |
| Decision Tree       |   94.21% |    90.16% | 91.67% |   90.91% |
| Random Forest       |   94.21% |    88.89% | 93.33% |   91.06% |
| SVM                 |   87.89% |    89.36% | 70.00% |   78.50% |

The Random Forest model was used for the Streamlit prediction interface.

## Model Insights

The Random Forest model provides feature importance values that are displayed in the Streamlit dashboard.

The project also includes DTI-based risk analysis to examine loan approval rates across different debt-to-income ratio groups.

These analyses are used to understand patterns within the dataset and support data-driven application analysis.

## Business Recommendations

The dashboard provides the following recommendations based on the analysis:

* Consider credit score and debt-to-income ratio when reviewing loan applications.
* Review applicant income, existing loans, and loan amount together to understand financial capacity.
* Use applicant demographics and employment information along with financial factors for application analysis.
* Use the machine learning prediction as a decision-support tool and review applications using appropriate financial criteria.
* Monitor approval patterns across different applicant groups to identify changes in loan approval outcomes.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook
* Git & GitHub

## Project Structure

```text
Loan_Approval_Prediction_System/
│
├── data/
│   └── loan_approval_data.csv
│
├── models/
│   ├── encoding_mappings.pkl
│   ├── model_metrics.csv
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── loan_approval_prediction.ipynb
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/mohammadyousuf-63/Loan_Approval_Prediction_System.git
```

### 2. Navigate to the project directory

```bash
cd Loan_Approval_Prediction_System
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install the required packages

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## Streamlit Application

The deployed application provides an interactive interface where users can enter applicant details and receive a loan approval prediction.

The dashboard also provides statistical analysis, feature importance, risk analysis, and business insights based on the dataset.

## Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Encoding & Scaling
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Random Forest Model
   ↓
Streamlit Prediction Interface
   ↓
Dashboard & Insights
```

## Conclusion

The Loan Approval Prediction System demonstrates a complete machine learning workflow for a classification problem. It covers data preprocessing, exploratory analysis, classification model development, evaluation, prediction, and dashboard-based analysis.

The Streamlit application provides an interactive way to enter applicant information, obtain a loan approval prediction, and explore loan approval patterns and insights from the dataset.

## Author

**Mohammad Yousuf**

GitHub: https://github.com/mohammadyousuf-63
