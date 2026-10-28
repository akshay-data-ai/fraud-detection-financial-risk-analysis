# PaySim Fraud Detection — Streamlit

Separate prototype for the full 6,362,620-row PaySim experiment. The original BIA Streamlit app at the repo root is unchanged.

## Required model (not included in source control)

Exported from the completed Colab training session:

`/content/drive/MyDrive/Fraud_Detection/Models/fraud_detection_random_forest.joblib`

Place this file at `paysim_app/models/fraud_detection_random_forest.joblib` **before deployment**. The bundle must include `model`, `feature_columns`, `type_mapping`, and `threshold`. The application uses the no-time-feature, no-SMOTE Random Forest and reproduces its original six-feature transformation.

Never load untrusted joblib/pickle files. Do not push the complete 493 MB PaySim CSV into GitHub.

## Setup

1. Add the trained model at the path above. The model needs to be accessible to the Streamlit server. For a public app, note that a public GitHub repository makes committed artifacts available to others.
2. Match the exact scikit-learn version from training. In Colab: `import sklearn; print(sklearn.__version__)`. Update this app's requirements version if necessary.
3. In the repo root: `pip install -r paysim_app/requirements.txt`
4. Run: `streamlit run paysim_app/app.py`
5. In Streamlit Community Cloud, select this repository and set main file path to `paysim_app/app.py`. The app uses its own `paysim_app/requirements.txt`.

## Features

- Overview of final held-out test metrics
- Individual simulated transaction scoring
- CSV batch scoring (10,000 rows maximum) and CSV download
- Explanation of features and model limitations

## Important

Synthetic PaySim data; results do not demonstrate readiness for real financial use. Scores are **uncalibrated** due to undersampling of legitimate training data. The fixed classification threshold is 0.5, as in the final Colab test. Pre-transaction fields only.

The full historical notebook and experiment reports remain in Google Drive. The prior BIA capstone dashboard in this repository is intentionally left intact.
