# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Ordinal mappings
MULTI_MAPPING = {
    'education_level': {'No formal': 0, 'Highschool': 1, 'Graduate': 2, 'Postgraduate': 3},
    'income_level': {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}
}

# Categorical columns to one-hot encode
MULTI_OHE = ['gender', 'ethnicity', 'employment_status', 'smoking_status']

# Numeric columns for MIN model
MIN_NUMERICAL_COLS = [
    'age',
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'sleep_hours_per_day',
    'screen_time_hours_per_day',
    'bmi',
    'systolic_bp',
    'diastolic_bp'
]

# Numeric columns for MAX model (includes labs)
MAX_NUMERICAL_COLS = [
    'age',
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'sleep_hours_per_day',
    'screen_time_hours_per_day',
    'bmi',
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'triglycerides',
    'insulin_level',
    'hba1c'
]

@st.cache_resource
def train_min_model():
    df = pd.read_csv("cleaned_diabetes_dataset_2.csv")

    y = df['diagnosed_diabetes']
    X = df.drop('diagnosed_diabetes', axis=1)

    min_model = [
        'age',
        "gender",
        "ethnicity",
        "education_level",
        "income_level",
        "employment_status",
        "smoking_status",
        "alcohol_consumption_per_week",
        "physical_activity_minutes_per_week",
        "sleep_hours_per_day",
        "screen_time_hours_per_day",
        "family_history_diabetes",
        "hypertension_history",
        "cardiovascular_history",
        "bmi",
        "systolic_bp",
        "diastolic_bp"
    ]

    X = X[min_model].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    multi_mapping = {
        'education_level': {'No formal': 0, 'Highschool': 1, 'Graduate': 2, 'Postgraduate': 3},
        'income_level': {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}
    }

    for col, mapping in multi_mapping.items():
        X_train[col] = X_train[col].map(mapping)
        X_test[col] = X_test[col].map(mapping)

    multi_OHE = ['gender', 'ethnicity', 'employment_status', 'smoking_status']

    X_train = pd.get_dummies(X_train, columns=multi_OHE, dtype=int)
    X_test = pd.get_dummies(X_test, columns=multi_OHE, dtype=int)

    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    min_numerical_cols = [
        'age',
        'alcohol_consumption_per_week',
        'physical_activity_minutes_per_week',
        'sleep_hours_per_day',
        'screen_time_hours_per_day',
        'bmi',
        'systolic_bp',
        'diastolic_bp'
    ]

    scaler = StandardScaler()
    scaler.fit(X_train[min_numerical_cols])

    X_train_scaled = X_train.copy()
    X_train_scaled[min_numerical_cols] = scaler.transform(X_train_scaled[min_numerical_cols])

    best_lr_min = LogisticRegression(
        C=0.001,
        penalty='l2',
        solver='lbfgs',
        class_weight=None,
        max_iter=5000
    )

    best_lr_min.fit(X_train_scaled, y_train)

    metadata = {
        "multi_mapping": multi_mapping,
        "multi_OHE": multi_OHE,
        "min_numerical_cols": min_numerical_cols,
        "train_columns": X_train_scaled.columns.tolist()
    }

    return best_lr_min, scaler, metadata

@st.cache_resource
def train_max_model():
    # Load dataset
    df = pd.read_csv("cleaned_diabetes_dataset_2.csv")

    # Target & features
    y = df['diagnosed_diabetes']
    X = df.drop('diagnosed_diabetes', axis=1)

    X_max = X.copy()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_max, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply ordinal mappings
    for col, mapping in MULTI_MAPPING.items():
        if col in X_train.columns:
            X_train[col] = X_train[col].map(mapping)
            X_test[col] = X_test[col].map(mapping)

    # One-hot encode
    X_train = pd.get_dummies(X_train, columns=MULTI_OHE, dtype=int)
    X_test = pd.get_dummies(X_test, columns=MULTI_OHE, dtype=int)

    # Align columns
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Scale numeric columns (MAX_NUMERICAL_COLS)
    scaler = StandardScaler()
    scaler.fit(X_train[MAX_NUMERICAL_COLS])

    X_train_scaled = X_train.copy()
    X_train_scaled[MAX_NUMERICAL_COLS] = scaler.transform(X_train_scaled[MAX_NUMERICAL_COLS])

    # Use best tuned hyperparameters for the MAX model
    best_lr_max = LogisticRegression(
        C=0.001,
        penalty='l2',
        solver='lbfgs',
        class_weight=None,
        max_iter=5000
    )

    best_lr_max.fit(X_train_scaled, y_train)

    # Save metadata needed for inference
    metadata = {
        "train_columns": X_train_scaled.columns.tolist()
    }

    return best_lr_max, scaler, metadata

def predict_diabetes_min(model, scaler, meta, user_input: dict, threshold: float = 0.35):
    train_columns = meta["train_columns"]

    row = pd.DataFrame([user_input])

    # Apply ordinal mapping
    for col, mapping in MULTI_MAPPING.items():
        row[col] = row[col].map(mapping)

    # One-hot encode categorical vars
    row = pd.get_dummies(row, columns=MULTI_OHE, dtype=int)

    # Align columns with training data - add missing dummies as 0
    row = row.reindex(columns=train_columns, fill_value=0)

    # Scale numeric columns
    row[MIN_NUMERICAL_COLS] = scaler.transform(row[MIN_NUMERICAL_COLS])

    # redict probability
    prob_diabetes = model.predict_proba(row)[:, 1][0]

    # Apply threshold
    pred_label = int(prob_diabetes >= threshold)

    return prob_diabetes, pred_label

def predict_diabetes_max(model, scaler, meta, user_input: dict, threshold: float = 0.35):
    train_columns = meta["train_columns"]

    row = pd.DataFrame([user_input])

    # Apply ordinal mappings
    for col, mapping in MULTI_MAPPING.items():
        if col in row.columns:
            row[col] = row[col].map(mapping)

    # One-hot encode categorical vars
    row = pd.get_dummies(row, columns=MULTI_OHE, dtype=int)

    #Align with training columns
    row = row.reindex(columns=train_columns, fill_value=0)

    # Scale numeric columns
    row[MAX_NUMERICAL_COLS] = scaler.transform(row[MAX_NUMERICAL_COLS])

    # Predict probability
    prob_diabetes = model.predict_proba(row)[:, 1][0]

    # Apply threshold
    pred_label = int(prob_diabetes >= threshold)

    return prob_diabetes, pred_label

def main():
    st.title("Diabetes Risk Prediction")
    st.write(
        "This app uses trained Logistic Regression models to estimate the probability "
        "of diabetes based on user inputs. You can switch between a **Minimal** model "
        "(no lab data) and a **Maximum** model (includes lab test values)."
    )

    # Train/load models
    with st.spinner("Training models..."):
        min_model, min_scaler, min_meta = train_min_model()
        max_model, max_scaler, max_meta = train_max_model()

    # Input form
    maximum_model = st.toggle("Advanced Model")

    st.header("Enter Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=120, value=50)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Asian", "Hispanic", "Other"])
        education_level = st.selectbox("Education Level", ["No formal", "Highschool", "Graduate", "Postgraduate"])
        income_level = st.selectbox("Income Level", ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"])
        employment_status = st.selectbox(
            "Employment Status",
            ["Employed", "Unemployed", "Retired", "Student", "Other"]
        )

    with col2:
        smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol = st.number_input(
            "Alcohol consumption per week (units)",
            min_value=0, max_value=100, value=1
        )
        physical_activity = st.number_input(
            "Physical activity (minutes per week)",
            min_value=0, max_value=2000, value=150
        )
        sleep_hours = st.number_input(
            "Sleep hours per day",
            min_value=0.0, max_value=24.0, value=7.0, step=0.5
        )
        screen_time = st.number_input(
            "Screen time hours per day",
            min_value=0.0, max_value=24.0, value=4.0, step=0.5
        )
        bmi = st.number_input(
            "BMI (kg/m^2)",
            min_value=10.0, max_value=60.0, value=28.0, step=0.1
        )
        systolic_bp = st.number_input(
            "Systolic BP (mmHg)",
            min_value=80, max_value=250, value=130
        )
        diastolic_bp = st.number_input(
            "Diastolic BP (mmHg)",
            min_value=40, max_value=150, value=85
        )

    st.subheader("Medical History")
    family_history = st.selectbox("Family history of diabetes", ["No", "Yes"])
    hypertension = st.selectbox("Hypertension history", ["No", "Yes"])
    cardiovascular = st.selectbox("Cardiovascular history", ["No", "Yes"])

    # Convert Yes/No to 0/1
    family_history_val = 1 if family_history == "Yes" else 0
    hypertension_val = 1 if hypertension == "Yes" else 0
    cardiovascular_val = 1 if cardiovascular == "Yes" else 0

    # Extra inputs for MAX model
    heart_rate = None
    hdl = None
    ldl = None
    trig = None
    insulin = None
    hba1c = None

    if maximum_model:
        st.subheader("Additional Lab Results (for Maximum model)")
        lab_col1, lab_col2 = st.columns(2)

        with lab_col1:
            heart_rate = st.number_input(
                "Heart rate (bpm)",
                min_value=30, max_value=200, value=75
            )
            hdl = st.number_input(
                "HDL cholesterol (mg/dL)",
                min_value=0, max_value=150, value=50
            )
            ldl = st.number_input(
                "LDL cholesterol (mg/dL)",
                min_value=0, max_value=300, value=130
            )

        with lab_col2:
            trig = st.number_input(
                "Triglycerides (mg/dL)",
                min_value=0, max_value=1000, value=180
            )
            insulin = st.number_input(
                "Insulin level (uU/mL)",
                min_value=0.0, max_value=300.0, value=10.0, step=0.1
            )
            hba1c = st.number_input(
                "HbA1c (%)",
                min_value=3.0, max_value=20.0, value=6.5, step=0.1
            )

    # Prediction
    if st.button("Predict Diabetes Risk"):
        # Base input for both models
        user_input = {
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity,
            "education_level": education_level,
            "income_level": income_level,
            "employment_status": employment_status,
            "smoking_status": smoking_status,
            "alcohol_consumption_per_week": alcohol,
            "physical_activity_minutes_per_week": physical_activity,
            "sleep_hours_per_day": sleep_hours,
            "screen_time_hours_per_day": screen_time,
            "family_history_diabetes": family_history_val,
            "hypertension_history": hypertension_val,
            "cardiovascular_history": cardiovascular_val,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp
        }

        if not maximum_model:
            prob, label = predict_diabetes_min(
                model=min_model,
                scaler=min_scaler,
                meta=min_meta,
                user_input=user_input
            )
            model_used = "Minimal (no lab tests)"

        else:
            # Add lab values for max model
            user_input.update({
                "heart_rate": heart_rate,
                "hdl_cholesterol": hdl,
                "ldl_cholesterol": ldl,
                "triglycerides": trig,
                "insulin_level": insulin,
                "hba1c": hba1c
            })

            prob, label = predict_diabetes_max(
                model=max_model,
                scaler=max_scaler,
                meta=max_meta,
                user_input=user_input
            )
            model_used = "Maximum (with lab tests)"


        # Output
        st.subheader("Prediction Results")
        st.write(f"**Model used:** {model_used}")
        st.write(f"**Predicted probability of diabetes:** {prob:.3f}")

        if label == 1:
            st.error("Model prediction: **Diabetes (at-risk)**")
        else:
            st.success("Model prediction: **No Diabetes**")

        st.caption(
            "Note: This is a model-based risk estimate and not a medical diagnosis. "
            "Clinical judgement and diagnostic tests should always be used."
        )

if __name__ == "__main__":
    main()
