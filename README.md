# California House Price Prediction

## Project Overview
This project predicts median house prices in California using the California Housing Dataset. It builds an end-to-end machine learning pipeline covering data preprocessing, feature engineering, model training, evaluation, and production-ready inference. The project follows real-world machine learning practices such as stratified sampling, preprocessing pipelines, cross-validation, and model persistence, making it suitable for data science and machine learning internship roles.

## Objectives
Predict median house values accurately
Reduce sampling bias using stratified sampling
Compare multiple regression models
Build a reusable and scalable machine learning pipeline
Ensure consistent preprocessing during training and inference

## Technologies and Libraries
Python
Pandas
NumPy
Scikit-learn
Joblib
Matplotlib
Seaborn

## Key Steps and Methodology

Stratified Sampling
Created an income category using the median_income feature
Applied StratifiedShuffleSplit to maintain the same income distribution in training and testing datasets
Improved model reliability by reducing sampling bias

## Data Preprocessing Pipeline
Built using Scikit-learn Pipelines and ColumnTransformer
Handled missing values using median imputation
Scaled numerical features using StandardScaler
Encoded categorical features using OneHotEncoder
Ensured automated and consistent preprocessing

## Model Training
Trained and compared Linear Regression
Decision Tree Regressor
Random Forest Regressor

## Model Evaluation
Used 10-fold cross-validation
Evaluation metric was Root Mean Squared Error
Random Forest performed best due to capturing non-linear patterns

## Production-Ready Inference
Saved trained model and preprocessing pipeline using Joblib
Applied the same pipeline to new input data
Generated predictions and exported results as a CSV file
Maintained consistency between training and inference

## Results and Insights
Random Forest outperformed Linear Regression and Decision Tree models
Cross-validation reduced overfitting
Pipeline-based design made the model scalable and deployment-ready

## Use Cases
Real estate price estimation
Data science and machine learning internship projects
Understanding regression problems
End-to-end machine learning pipeline practice

## Project Highlights
Complete end-to-end machine learning workflow
Industry-standard preprocessing pipeline
Proper model comparison and evaluation
Clean and reusable code structure

## 👤 Author

Sanil Gupta  
BCA (AI & ML) Student  
Aspiring Data Scientist

GitHub: https://github.com/Sanil656  
LinkedIn: https://www.linkedin.com/in/sanil-gupta-64bb11314/
