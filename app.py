import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --------------- Configurations ---------------
FEATURE_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# --------------- Page Title ---------------
st.title("💳 Credit Card Fraud Detection Dashboard")

# --------------- Sidebar Navigation ---------------
menu = ["Home", "Model Metrics", "Predict Fraud", "Manual Prediction"]
choice = st.sidebar.selectbox("📊 Navigate", menu)

# --------------- Home Page ---------------
if choice == "Home":
    st.write("""
    ## 🧐 Project Overview
    Welcome to the Credit Card Fraud Detection app!  
    - Built with Python, Streamlit, and scikit-learn.  
    - Uses logistic regression on Kaggle's anonymized data.  
    - Handles class imbalance via under-sampling.  
    - Presents interactive dashboards and prediction interfaces.

    **Use the sidebar to view model performance or make predictions with your own data.**
    """)
    with st.expander("About the Data & FAQ"):
        st.write("""
        - **Time**: Seconds since first transaction.
        - **V1–V28**: PCA-transformed features for privacy.
        - **Amount**: Transaction amount.
        - **Class**: Target (0=Legit, 1=Fraud).

        **FAQ:**
        - This demo uses anonymized data and does not expose real card details.
        - Uploads and manual predictions require all feature columns, not the 'Class'.
        """)

# --------------- Model Metrics Page ---------------
elif choice == "Model Metrics":
    st.header("📈 Model Performance")

    # Evaluation Data
    data = pd.read_csv("creditcard.csv")
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    legit_sample = legit.sample(n=492, random_state=42)
    test_data = pd.concat([legit_sample, fraud])

    X = test_data[FEATURE_COLUMNS]
    y = test_data['Class']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("### 🧾 Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve & Area
    st.write("### ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    fig_roc = go.Figure(
        data=[go.Scatter(x=fpr, y=tpr, name='ROC Curve')],
        layout=go.Layout(title=f'ROC Curve (AUC={roc_auc:.2f})', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    )
    st.plotly_chart(fig_roc)

    # Precision-Recall Curve
    st.write("### Precision-Recall Curve")
    prec, rec, pr_thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    fig_pr = go.Figure(
        data=[go.Scatter(x=rec, y=prec, name='Precision-Recall')],
        layout=go.Layout(title=f'Precision-Recall Curve (AUC={pr_auc:.2f})', xaxis_title='Recall', yaxis_title='Precision')
    )
    st.plotly_chart(fig_pr)

    # Feature Importance (Logistic Regression Coefs)
    if st.checkbox("Show Feature Importance"):
        importances = np.abs(model.coef_[0])
        feat_imp = pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=False)
        st.write("### Feature Importance (abs(Logistic Coefficient))")
        st.bar_chart(feat_imp)

# --------------- Predict Fraud Page ---------------
elif choice == "Predict Fraud":
    st.header("🚨 Predict New Transactions")

    st.markdown("""
    - Upload a CSV file:  
       
    
    """)

    st.download_button(
        label="Download Demo Sample Input",
        data=pd.DataFrame([np.zeros(len(FEATURE_COLUMNS))], columns=FEATURE_COLUMNS).to_csv(index=False),
        file_name="sample_transaction.csv",
        mime="text/csv"
    )

    file = st.file_uploader("Upload CSV file with transactions", type=['csv'])
    if file is not None:
        try:
            user_data = pd.read_csv(file)
            # Drop 'Class' if present
            if 'Class' in user_data.columns:
                user_data = user_data.drop(columns=['Class'])

            # Validate columns
            if sorted(user_data.columns) != sorted(FEATURE_COLUMNS):
                st.error(f"Incorrect columns. File must contain exactly: {FEATURE_COLUMNS}")
            else:
                input_scaled = scaler.transform(user_data[FEATURE_COLUMNS])
                predictions = model.predict(input_scaled)
                probs = model.predict_proba(input_scaled)[:, 1]
                user_data["Prediction"] = predictions
                user_data["Fraud Probability"] = probs
                st.write("### 🧾 Prediction Results")
                st.dataframe(user_data)
                st.download_button("Download Prediction Results", user_data.to_csv(index=False), "predictions.csv")
                st.success("0 = Legit ✅ | 1 = Fraud ❗")
        except Exception as e:
            st.error(f"Error processing your file: {e}")
    else:
        st.info("Awaiting CSV upload.")

# --------------- Manual Prediction Page (Single Transaction) ---------------
elif choice == "Manual Prediction":
    st.header("📝 Manual Transaction Entry")
    st.write("Enter transaction details:")

    manual_input = {}
    for col in FEATURE_COLUMNS:
        manual_input[col] = st.number_input(f"{col}", value=0.0)
    manual_row = pd.DataFrame([manual_input])
    if st.button("Predict"):
        scaled = scaler.transform(manual_row)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0, 1]
        st.write(f"### Prediction: {'Fraud ❗' if pred == 1 else 'Legit ✅'}")
        st.write(f"Fraud Probability: {prob:.4f}")
        if pred == 1:
            st.balloons()
