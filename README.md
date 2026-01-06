# Task 2 part 1: End-to-End ML Pipeline for Customer Churn Prediction

## 📌 Problem Statement & Objective
The objective of this task is to build a robust, production-ready machine learning pipeline to predict customer churn (whether a customer will leave or stay). We utilized the Telco Customer Churn dataset and implemented a Scikit-learn pipeline to automate data preprocessing, feature scaling, and model inference.

## ⚙️ Methodology / Approach
1.  **Data Loading:** Loaded the Telco Customer Churn dataset directly from a public repository.
2.  **Preprocessing:**
    * Converted 'TotalCharges' to numeric and handled missing values.
    * Built a `ColumnTransformer` to apply `StandardScaler` to numeric features and `OneHotEncoder` to categorical features automatically.
3.  **Pipeline Construction:** Wrapped the preprocessor and a `RandomForestClassifier` into a single Scikit-learn `Pipeline` object.
4.  **Hyperparameter Tuning:** Used `GridSearchCV` to find the best `n_estimators` and `max_depth` for the Random Forest model.
5.  **Export:** Saved the final trained pipeline as `churn_pipeline.pkl` using Joblib for future reuse.

## 📊 Key Results
* **Best Model Parameters:** Identified via GridSearch.
* **Accuracy:** Achieved ~80% accuracy on the test set.
* **Confusion Matrix:** Generated to visualize True Positives vs False Negatives.

## 🛠 Tools Used
* Python (Pandas, NumPy)
* Scikit-learn (Pipeline, GridSearchCV, RandomForest)
* Joblib (Model serialization)
