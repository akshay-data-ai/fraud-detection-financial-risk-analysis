
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Risk Analytics",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("fraud_detection_pipeline.pkl")
    config = joblib.load("model_config.pkl")
    return model, config


@st.cache_data
def load_dashboard_data():
    model_comparison = pd.read_csv("model_comparison.csv")
    threshold_analysis = pd.read_csv("threshold_analysis.csv")
    feature_importance = pd.read_csv("feature_importance.csv")
    financial_summary = pd.read_csv("financial_summary.csv")

    return (
        model_comparison,
        threshold_analysis,
        feature_importance,
        financial_summary
    )


model, config = load_model()

(
    model_comparison,
    threshold_analysis,
    feature_importance,
    financial_summary
) = load_dashboard_data()

FINAL_THRESHOLD = config["threshold"]


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ Fraud Risk Analytics")

st.markdown(
    """
    ### Machine Learning Based Mobile Transaction Fraud Detection

    Interactive fraud-risk prototype for transaction classification,
    model analytics and financial-impact assessment.
    """
)

st.info(
    "This application is a capstone prototype developed using "
    "simulated transaction data. Predictions should not be interpreted "
    "as real banking decisions."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Fraud Risk Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Fraud Prediction",
        "Model Analytics",
        "Financial Risk"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Model Information")
st.sidebar.write(f"**Model:** {config['model_name']}")
st.sidebar.write(f"**Threshold:** {FINAL_THRESHOLD:.2f}")
st.sidebar.write("**Application:** Fraud Detection Prototype")


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.header("Executive Dashboard")

    st.write(
        "Summary of fraud detection performance and financial exposure "
        "on the holdout test dataset."
    )

    # Holdout confusion matrix:
    # TN = 2001
    # FP = 0
    # FN = 1
    # TP = 227

    tn = 2001
    fp = 0
    fn = 1
    tp = 227

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    col3.metric(
        "Fraud Recall",
        f"{recall * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Frauds Detected",
        "227 / 228"
    )

    col2.metric(
        "False Positives",
        "0"
    )

    col3.metric(
        "Missed Frauds",
        "1"
    )

    col4.metric(
        "Fraud Value Detected",
        "99.82%"
    )

    st.divider()

    st.subheader("Model Comparison")

    st.dataframe(
        model_comparison.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Top Fraud Risk Drivers")

    top_features = feature_importance.head(10).copy()

    top_features["Feature"] = (
        top_features["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    chart_data = top_features.set_index(
        "Feature"
    )["Importance"]

    st.bar_chart(chart_data)

    st.caption(
        "Feature importance is derived from the selected Random Forest "
        "model. The simulated dataset contains strong temporal patterns, "
        "so feature importance should not automatically be generalized "
        "to real banking data."
    )


# ============================================================
# FRAUD PREDICTION PAGE
# ============================================================

elif page == "Fraud Prediction":

    st.header("Transaction Fraud Prediction")

    st.write(
        "Enter transaction details to estimate fraud probability "
        "using the trained Random Forest pipeline."
    )

    col1, col2 = st.columns(2)

    with col1:

        step = st.number_input(
            "Transaction Step",
            min_value=0,
            value=1,
            step=1
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "CASH_IN",
                "CASH_OUT",
                "DEBIT",
                "PAYMENT",
                "TRANSFER"
            ]
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0,
            format="%.2f"
        )

        oldbalanceOrg = st.number_input(
            "Origin Account - Old Balance",
            min_value=0.0,
            value=5000.0,
            format="%.2f"
        )

        newbalanceOrig = st.number_input(
            "Origin Account - New Balance",
            min_value=0.0,
            value=4000.0,
            format="%.2f"
        )

    with col2:

        oldbalanceDest = st.number_input(
            "Destination - Old Balance",
            min_value=0.0,
            value=0.0,
            format="%.2f"
        )

        newbalanceDest = st.number_input(
            "Destination - New Balance",
            min_value=0.0,
            value=1000.0,
            format="%.2f"
        )

        merchant_destination = st.selectbox(
            "Merchant Destination?",
            ["No", "Yes"]
        )

    if st.button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------------

        hour_in_cycle = step % 24

        log_amount = np.log1p(amount)

        origin_balance_change = (
            oldbalanceOrg - newbalanceOrig
        )

        origin_balance_error = (
            oldbalanceOrg
            - amount
            - newbalanceOrig
        )

        destination_balance_change = (
            newbalanceDest
            - oldbalanceDest
        )

        destination_balance_error = (
            oldbalanceDest
            + amount
            - newbalanceDest
        )

        merchant_flag = (
            1 if merchant_destination == "Yes" else 0
        )

        origin_new_balance_zero = (
            1 if newbalanceOrig == 0 else 0
        )

        destination_old_balance_zero = (
            1 if oldbalanceDest == 0 else 0
        )

        if oldbalanceOrg > 0:
            amount_to_oldbalance_ratio = (
                amount / oldbalanceOrg
            )
        else:
            amount_to_oldbalance_ratio = 0


        # ----------------------------------------
        # CREATE MODEL INPUT
        # ----------------------------------------

        input_data = pd.DataFrame({

            "step": [step],

            "type": [transaction_type],

            "amount": [amount],

            "oldbalanceOrg": [
                oldbalanceOrg
            ],

            "newbalanceOrig": [
                newbalanceOrig
            ],

            "oldbalanceDest": [
                oldbalanceDest
            ],

            "newbalanceDest": [
                newbalanceDest
            ],

            "hour_in_cycle": [
                hour_in_cycle
            ],

            "log_amount": [
                log_amount
            ],

            "origin_balance_change": [
                origin_balance_change
            ],

            "origin_balance_error": [
                origin_balance_error
            ],

            "destination_balance_change": [
                destination_balance_change
            ],

            "destination_balance_error": [
                destination_balance_error
            ],

            "is_merchant_dest": [
                merchant_flag
            ],

            "origin_new_balance_zero": [
                origin_new_balance_zero
            ],

            "destination_old_balance_zero": [
                destination_old_balance_zero
            ],

            "amount_to_oldbalance_ratio": [
                amount_to_oldbalance_ratio
            ]
        })


        # ----------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------

        fraud_probability = (
            model.predict_proba(
                input_data
            )[0, 1]
        )

        prediction = int(
            fraud_probability >= FINAL_THRESHOLD
        )


        # ----------------------------------------
        # RISK BAND
        # ----------------------------------------

        if fraud_probability >= 0.75:

            risk_level = "High"

        elif fraud_probability >= 0.50:

            risk_level = "Elevated"

        elif fraud_probability >= 0.20:

            risk_level = "Moderate"

        else:

            risk_level = "Low"


        # ----------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------

        st.divider()

        st.subheader("Risk Assessment")

        col1, col2 = st.columns(2)

        col1.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )

        col2.metric(
            "Risk Band",
            risk_level
        )

        st.progress(
            min(
                max(
                    float(fraud_probability),
                    0.0
                ),
                1.0
            )
        )

        if prediction == 1:

            st.error(
                "⚠️ Transaction classified as "
                "**POTENTIALLY FRAUDULENT**"
            )

            st.write(
                "The predicted fraud probability exceeds the "
                f"{FINAL_THRESHOLD:.2f} operating threshold."
            )

        else:

            st.success(
                "✓ Transaction classified as "
                "**LEGITIMATE**"
            )

            st.write(
                "The predicted fraud probability is below the "
                f"{FINAL_THRESHOLD:.2f} operating threshold."
            )

        with st.expander(
            "View engineered model features"
        ):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

    st.caption(
        "Risk bands are presentation aids and are not separately "
        "trained risk classifications."
    )


# ============================================================
# MODEL ANALYTICS
# ============================================================

elif page == "Model Analytics":

    st.header("Model Analytics")

    st.subheader("Model Performance Comparison")

    st.dataframe(
        model_comparison.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("F1 Score Comparison")

    chart_df = model_comparison.copy()

    chart_df["Model Method"] = (
        chart_df["Model"]
        + " - "
        + chart_df["Sampling"]
    )

    st.bar_chart(
        chart_df.set_index(
            "Model Method"
        )["F1"]
    )

    st.divider()

    st.subheader("Random Forest Feature Importance")

    importance_display = (
        feature_importance
        .head(15)
        .copy()
    )

    importance_display["Feature"] = (
        importance_display["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    st.bar_chart(
        importance_display.set_index(
            "Feature"
        )["Importance"]
    )

    with st.expander(
        "View feature importance table"
    ):

        st.dataframe(
            importance_display,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader("Threshold Sensitivity Analysis")

    st.write(
        """
        A lower classification threshold increases sensitivity
        to fraud but may also increase false-positive alerts.
        The final operating threshold was retained at 0.50 after
        evaluating this trade-off.
        """
    )

    threshold_display = threshold_analysis.copy()

    st.dataframe(
        threshold_display.round(4),
        use_container_width=True,
        hide_index=True
    )

    if (
        "Threshold" in threshold_display.columns
        and "F1" in threshold_display.columns
    ):

        threshold_chart = (
            threshold_display[
                [
                    "Threshold",
                    "Precision",
                    "Recall",
                    "F1"
                ]
            ]
            .set_index("Threshold")
        )

        st.line_chart(
            threshold_chart
        )


# ============================================================
# FINANCIAL RISK
# ============================================================

elif page == "Financial Risk":

    st.header("Financial Risk Analysis")

    st.write(
        "Financial exposure based on the holdout test dataset."
    )

    actual_fraud_value = 235846526.37
    detected_fraud_value = 235430525.04
    missed_fraud_value = 416001.33

    value_detection_rate = (
        detected_fraud_value
        / actual_fraud_value
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Actual Fraud Value",
        f"{actual_fraud_value:,.2f}"
    )

    col2.metric(
        "Detected Fraud Value",
        f"{detected_fraud_value:,.2f}"
    )

    col3.metric(
        "Missed Fraud Exposure",
        f"{missed_fraud_value:,.2f}"
    )

    col4.metric(
        "Value Detection Rate",
        f"{value_detection_rate * 100:.2f}%"
    )

    st.caption(
        "All monetary values are expressed in local currency units "
        "because the dataset does not specify a currency."
    )

    st.divider()

    st.subheader("Financial Scenario Simulator")

    st.write(
        """
        Adjust the assumptions below to explore the potential
        financial effect of different recovery rates and
        investigation costs.
        """
    )

    recovery_rate = (
        st.slider(
            "Fraud Prevention / Recovery Rate (%)",
            min_value=0,
            max_value=100,
            value=80,
            step=5
        )
        / 100
    )

    investigation_cost_per_alert = (
        st.number_input(
            "Investigation Cost per Alert",
            min_value=0.0,
            value=50.0,
            step=10.0
        )
    )

    number_of_alerts = 227

    estimated_prevented_loss = (
        detected_fraud_value
        * recovery_rate
    )

    total_investigation_cost = (
        number_of_alerts
        * investigation_cost_per_alert
    )

    estimated_net_benefit = (
        estimated_prevented_loss
        - total_investigation_cost
    )

    remaining_fraud_loss = (
        actual_fraud_value
        - estimated_prevented_loss
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Estimated Prevented Loss",
        f"{estimated_prevented_loss:,.2f}"
    )

    col2.metric(
        "Estimated Investigation Cost",
        f"{total_investigation_cost:,.2f}"
    )

    col3, col4 = st.columns(2)

    col3.metric(
        "Remaining Fraud Loss",
        f"{remaining_fraud_loss:,.2f}"
    )

    col4.metric(
        "Estimated Net Benefit",
        f"{estimated_net_benefit:,.2f}"
    )

    st.warning(
        "The recovery rate and investigation cost are scenario "
        "assumptions. They are not supplied by the original dataset "
        "and should not be interpreted as actual bank costs."
    )

    st.divider()

    st.subheader("Financial Analysis Output")

    st.dataframe(
        financial_summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fraud Detection & Financial Risk Analysis | "
    "Data Science & AI Capstone Project"
)
