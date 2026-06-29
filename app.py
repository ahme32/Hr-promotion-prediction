# HR Analytics - Employee Promotion Prediction
# Simple Streamlit web app that loads the saved model and predicts promotion.

import streamlit as st
import pandas as pd
import joblib


# Load the saved model, scaler, and feature list once (cached so it is fast)
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features


model, scaler, features = load_artifacts()


# Page title and short description
st.title("Employee Promotion Prediction")
st.write("Fill in the employee details below and click Predict to see if they are likely to be promoted.")


# --- User inputs ---

# Categorical inputs (the options are the real categories from the dataset)
department = st.selectbox(
    "Department",
    ["Sales & Marketing", "Operations", "Technology", "Analytics", "R&D",
     "Procurement", "Finance", "HR", "Legal"],
)

education = st.selectbox(
    "Education",
    ["Bachelor's", "Master's & above", "Below Secondary"],
)

# Show a friendly label, then convert it to the value used in training ('f' or 'm')
gender_choice = st.selectbox("Gender", ["Female", "Male"])
gender = "m" if gender_choice == "Male" else "f"

recruitment_channel = st.selectbox(
    "Recruitment channel",
    ["sourcing", "other", "referred"],
)

# Show a friendly label, then convert it to 1 or 0
awards_choice = st.selectbox("Won an award last year?", ["No", "Yes"])
awards_won = 1 if awards_choice == "Yes" else 0

# Numeric inputs
no_of_trainings = st.number_input("Number of trainings", min_value=1, max_value=15, value=1, step=1)
age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1)
previous_year_rating = st.slider("Previous year rating (1 to 5)", min_value=1, max_value=5, value=3)
length_of_service = st.number_input("Length of service (years)", min_value=1, max_value=40, value=5, step=1)
avg_training_score = st.number_input("Average training score", min_value=30, max_value=100, value=60, step=1)


# --- Build one row of features that matches the training columns exactly ---
def build_feature_row():
    # Start with a row of zeros that has the exact training feature columns
    row = pd.DataFrame([[0] * len(features)], columns=features)

    # Fill the numeric features
    row.at[0, "no_of_trainings"] = no_of_trainings
    row.at[0, "age"] = age
    row.at[0, "previous_year_rating"] = previous_year_rating
    row.at[0, "length_of_service"] = length_of_service
    row.at[0, "awards_won"] = awards_won
    row.at[0, "avg_training_score"] = avg_training_score

    # Create the engineered features using the same formulas as in the notebook
    row.at[0, "total_score"] = avg_training_score * previous_year_rating
    row.at[0, "high_training_score"] = 1 if avg_training_score >= 80 else 0

    # For each category, turn on its matching dummy column if it exists.
    # The reference category (dropped during training) simply stays all zeros.
    category_inputs = [
        ("department", department),
        ("education", education),
        ("gender", gender),
        ("recruitment_channel", recruitment_channel),
    ]
    for column_name, chosen_value in category_inputs:
        dummy_column = column_name + "_" + str(chosen_value)
        if dummy_column in features:
            row.at[0, dummy_column] = 1

    return row


# --- Predict when the button is clicked ---
if st.button("Predict"):
    # Build the input row
    input_row = build_feature_row()

    # Scale the input the same way the training data was scaled
    input_scaled = scaler.transform(input_row)

    # Get the prediction (0 or 1) and the probability of promotion
    prediction = model.predict(input_scaled)[0]
    promotion_probability = model.predict_proba(input_scaled)[0][1]

    # Show the result to the user
    if prediction == 1:
        st.success("This employee is likely to be PROMOTED.")
    else:
        st.error("This employee is NOT likely to be promoted.")

    st.write("Probability of promotion:", round(promotion_probability * 100, 1), "%")
