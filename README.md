# Fraud Detection & Financial Risk Analysis for Mobile Transactions

### Machine Learning | Python | Scikit-learn | Streamlit

An end-to-end Data Science and Artificial Intelligence capstone project focused on detecting potentially fraudulent mobile money transactions and evaluating their associated financial risk.

**Live Dashboard:** https://akshay-fraud-risk-analytics.streamlit.app/

## 1. Project Overview

Financial fraud creates significant operational and financial challenges for banks and payment service providers.

This project demonstrates how machine learning can support transaction risk assessment by identifying suspicious transaction patterns, comparing classification models, and estimating potential financial exposure.

The solution includes an interactive Streamlit application for transaction-level predictions, model performance analysis, and financial risk simulation.

## 2. Dataset

The project uses a simulated mobile money transaction dataset.

| Dataset Attribute | Value |
|---|---|
| Total transactions | 11,142 |
| Legitimate transactions | 10,000 |
| Fraudulent transactions | 1,142 |
| Target variable | `isFraud` |
| Dataset file | `Fraud_Analysis_Dataset.csv` |

The dataset includes transaction type, amount, transaction step, and origin/destination account balances.

## 3. Project Workflow

1. Data loading and quality assessment
2. Exploratory data analysis (EDA)
3. Feature engineering and preprocessing
4. Class imbalance handling
5. Machine learning model development
6. Model evaluation and comparison
7. Fraud probability and threshold analysis
8. Financial impact simulation
9. Interactive Streamlit dashboard deployment

## 4. Machine Learning Models

The following approaches were evaluated:

- Logistic Regression
- Logistic Regression with class weighting
- Logistic Regression with SMOTE
- Random Forest
- Random Forest with class weighting
- Gradient Boosting

The final deployed model is a **Random Forest classifier**, using an operating classification threshold of **0.50**.

## 5. Final Model Performance

Performance on the held-out test dataset:

| Metric | Result |
|---|---:|
| Accuracy | 99.96% |
| Precision | 100.00% |
| Recall | 99.56% |
| F1 Score | 99.78% |
| ROC-AUC | 99.9964% |
| PR-AUC | 99.9679% |

### Confusion Matrix Results

| Outcome | Transactions |
|---|---:|
| True Positives | 227 |
| True Negatives | 2,001 |
| False Positives | 0 |
| False Negatives | 1 |

These results reflect evaluation on the supplied simulated dataset and should not be interpreted as guaranteed performance in a live banking environment.

## 6. Financial Risk Analysis

The financial analysis evaluates the value of fraudulent transactions identified by the final model.

| Financial Metric | Value |
|---|---:|
| Actual fraud value | 235,846,526.37 |
| Detected fraud value | 235,430,525.04 |
| Missed fraud exposure | 416,001.33 |
| Fraud value detection rate | 99.82% |

The dataset does not establish a specific currency, so monetary amounts are presented in dataset units.

The application also includes an illustrative financial benefit simulator with adjustable recovery assumptions and investigation costs. These estimates are hypothetical and do not represent realized savings.

## 7. Streamlit Application

**Launch the dashboard:** https://akshay-fraud-risk-analytics.streamlit.app/

The application contains four sections:

- **Executive Dashboard:** Summary of model performance and fraud risk.
- **Fraud Prediction:** Transaction-level fraud probability, classification, and risk assessment.
- **Model Analytics:** Model comparison and feature importance visualizations.
- **Financial Risk:** Fraud exposure metrics and a scenario-based financial simulator.

## 8. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn
- Plotly
- Streamlit
- Google Colab
- Git and GitHub

## 9. Repository Files

| File | Description |
|---|---|
| `Fraud_Detection_Akshay_Timbadiya_FINAL.ipynb` | Final machine learning notebook |
| `Fraud_Analysis_Dataset.csv` | Simulated transaction dataset |
| `app.py` | Streamlit dashboard application |
| `fraud_detection_pipeline.pkl` | Exported trained model pipeline |
| `model_config.pkl` | Model configuration |
| `model_comparison.csv` | Model evaluation results |
| `feature_importance.csv` | Feature importance results |
| `threshold_analysis.csv` | Classification threshold analysis |
| `financial_summary.csv` | Financial risk analysis results |
| `requirements.txt` | Python dependencies |

## 10. How to Run

### Run the Machine Learning Notebook

1. Open Google Colab.
2. Upload `Fraud_Detection_Akshay_Timbadiya_FINAL.ipynb`.
3. Upload `Fraud_Analysis_Dataset.csv` to the Colab session.
4. Run all notebook cells in sequence.
5. Review the model evaluation, confusion matrix, feature importance, and financial analysis outputs.

### Run the Streamlit Application Locally

Clone the repository:

```bash
git clone https://github.com/akshay-data-ai/fraud-detection-financial-risk-analysis.git
cd fraud-detection-financial-risk-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

The exported model was created using scikit-learn 1.6.1; use the repository's pinned dependencies to maintain compatibility.

## 11. Project Limitations

- The dataset is simulated and may not reflect the complexity of real-world banking transactions.
- Evaluation uses a random stratified train/test split rather than time-based validation.
- Temporal features such as `step` and `hour_in_cycle` are highly influential and may reflect simulation-specific patterns.
- Classification threshold exploration on the test set may produce optimistic estimates.
- Financial benefit calculations depend on assumed recovery rates and investigation costs.
- Production deployment would require additional validation, monitoring, governance, and security controls.

## 12. Conclusion

This project demonstrates an end-to-end machine learning workflow for mobile transaction fraud detection, from data preparation and model development to risk visualization and financial impact assessment.

The final Random Forest model achieved strong results on the simulated holdout dataset, while the deployed Streamlit dashboard provides an accessible prototype for exploring transaction risk and model performance.

---

**Project Type:** Data Science & AI Capstone Project  
**Institute:** Boston Institute of Analytics  
**Author:** Akshay Timbadiya  
**GitHub:** https://github.com/akshay-data-ai
