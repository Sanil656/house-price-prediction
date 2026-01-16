# house-pricr-prediction-

## California Housing Price Prediction using Machine Learning

This project focuses on predicting median house prices in California using the California Housing Dataset. The goal is to build an end-to-end machine learning pipeline that handles data preprocessing, feature engineering, model training, evaluation, and deployment-ready inference.

The project follows real-world ML practices, including stratified sampling, pipelines, cross-validation, and model persistence.

## Technologies & Libraries
Python
Pandas, NumPy
Scikit-learn
Joblib
Matplotlib / Seaborn (for EDA, if used)

## Key Steps & Methodology

1️⃣ Stratified Sampling
Created an income category using median_income
Applied StratifiedShuffleSplit to ensure train/test sets maintain the same income distribution
Prevents sampling bias and improves model reliability

2️⃣ Data Preprocessing Pipeline
Used Scikit-learn Pipelines and ColumnTransformer:
Missing values handled using median imputation
Feature scaling using StandardScaler
Categorical Features
Handled using OneHotEncoder

3️⃣ Model Training
Trained and compared multiple regression models:
Linear Regression
Decision Tree Regressor
Random Forest Regressor

4️⃣ Model Evaluation
Used 10-fold Cross-Validation
Evaluation metric: Root Mean Squared Error (RMSE)
Random Forest achieved the best performance due to its ability to capture non-linear patterns

5️⃣ Production-Ready Inference Pipeline
Trained model and preprocessing pipeline saved using Joblib
New input data is transformed using the same pipeline
Predictions are generated and exported as a CSV file
Ensures consistent preprocessing during training and inference

📈 Results
Random Forest outperformed Linear Regression and Decision Tree models
Cross-validation helped prevent overfitting
Pipeline-based approach made the model scalable and deployment-ready

🔗 Use Cases
Real estate price estimation
Data science internship projects
ML pipeline and deployment practice
Regression problem understanding
