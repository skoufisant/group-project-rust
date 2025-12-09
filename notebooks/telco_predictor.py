import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path
    import joblib
    import marimo as mo
    import pandas as pd


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Telco Churn Predictor

    Adjust the customer features below and see the churn prediction in real-time.
    """)
    return


@app.cell
def _():
    # Load the model and scaler
    BUNDLE = joblib.load("models/telco_logistic_regression.joblib")
    MODEL = BUNDLE["model"]
    SCALER = BUNDLE["scaler"]

    FEATURE_ORDER = ["tenure", "MonthlyCharges", "TechSupport_yes", "PhoneService_yes", "Contract_one_year", "Contract_two_year", "InternetService_fiber_optic", "InternetService_no"]

    return FEATURE_ORDER, MODEL, SCALER


@app.cell
def _(mo):
    # Input controls - numeric sliders and checkboxes
    tenure = mo.ui.slider(0, 72, label="Tenure (months)", value=12)
    monthly_charges = mo.ui.slider(0, 150, label="Monthly Charges ($)", value=65)
    tech_support = mo.ui.checkbox(label="Tech Support")
    phone_service = mo.ui.checkbox(label="Phone Service")
    
    mo.md(f"""
    ## Customer Information

    {tenure}

    {monthly_charges}

    {tech_support}

    {phone_service}
    """)

    return tenure, monthly_charges, tech_support, phone_service


@app.cell
def _(mo):
    # Contract type radio - SEPARATE CELL
    contract = mo.ui.radio(
        options={"month_to_month": "Month-to-Month", "one_year": "One Year", "two_year": "Two Year"},
        value="month_to_month",
        label="Contract Type"
    )
    
    mo.md(f"""
    ## Contract Details

    {contract}
    """)

    return (contract,)


@app.cell
def _(mo):
    # Internet service radio - SEPARATE CELL
    internet = mo.ui.radio(
        options={"no": "No Internet", "fiber": "Fiber Optic", "dsl": "DSL"},
        value="dsl",
        label="Internet Service"
    )
    
    mo.md(f"""
    ## Internet Service

    {internet}
    """)

    return (internet,)


@app.cell
def _(
    tenure,
    monthly_charges,
    tech_support,
    phone_service,
    contract,
    internet,
    MODEL,
    SCALER,
    FEATURE_ORDER,
    pd,
):
    # Access values to trigger reactivity
    t_val = tenure.value
    m_val = monthly_charges.value
    ts_val = tech_support.value
    ps_val = phone_service.value
    c_val = contract.value
    i_val = internet.value
    
    # Build feature vector
    features_dict = {
        "tenure": t_val,
        "MonthlyCharges": m_val,
        "TechSupport_yes": 1 if ts_val else 0,
        "PhoneService_yes": 1 if ps_val else 0,
        "Contract_one_year": 1 if c_val == "one_year" else 0,
        "Contract_two_year": 1 if c_val == "two_year" else 0,
        "InternetService_fiber_optic": 1 if i_val == "fiber" else 0,
        "InternetService_no": 1 if i_val == "no" else 0,
    }

    # Create feature array in correct order
    feature_values = [features_dict[feature] for feature in FEATURE_ORDER]
    features_df = pd.DataFrame([feature_values], columns=FEATURE_ORDER)

    # Scale and predict
    scaled_features = SCALER.transform(features_df)
    churn_probability = float(MODEL.predict_proba(scaled_features)[0, 1])

    return churn_probability, features_dict


@app.cell(hide_code=True)
def _(churn_probability, features_dict, mo):
    # Display prediction
    percentage = churn_probability * 100

    if churn_probability < 0.3:
        risk_level = "🟢 Low Risk"
    elif churn_probability < 0.6:
        risk_level = "🟡 Medium Risk"
    else:
        risk_level = "🔴 High Risk"

    mo.md(f"""
    ## Prediction Result

    **Churn Probability: {percentage:.1f}%**

    **Risk Level: {risk_level}**

    ### Current Inputs
    ```
    {features_dict}
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
