#Health Insurance Claim Prediction
#Overview
This project aims to predict health insurance claims using a dataset containing various features related to individuals' health and demographics. Multiple machine learning models are used to predict the claim amount based on these features, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, and XGBoost Regressor. The models are evaluated on their performance across different cities to identify the best-performing model and to understand which model works best in different contexts.
Data
The dataset used is 1651277648862_healthinsurance.csv, which contains the following columns:
age: Age of the individual
sex: Gender of the individual
weight: Weight of the individual
bmi: Body Mass Index of the individual
hereditary_diseases: Whether the individual has hereditary diseases
no_of_dependents: Number of dependents
smoker: Whether the individual is a smoker
city: City where the individual resides
bloodpressure: Blood pressure of the individual
diabetes: Whether the individual has diabetes
regular_ex: Whether the individual has regular exercise
job_title: Job title of the individual
claim: Amount of the health insurance claim (target variable)
Methodology
Data Preprocessing:
Handled missing values by filling numerical columns with the mean and categorical columns with the mode.
Encoded categorical variables using LabelEncoder.
Feature and Target Variable:
Features (X) include all columns except claim.
Target variable (y) is the claim column.
Data Splitting:
Split the data into training and testing sets with an 80-20 split.
Standardization:
Standardized features using StandardScaler to ensure that all features contribute equally to the model.
Model Training and Evaluation:
Trained four different models:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
XGBoost Regressor
Evaluated models using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) metrics.
City-wise Analysis:
Trained and evaluated models separately for each city.
Created a summary of model performance across the top 10 cities.
Results
Overall Model Performance
Linear Regression:
MSE: 38,248,652.46
RMSE: 6,184.55
R2: 0.744
Decision Tree Regressor:
MSE: 7,232,083.29
RMSE: 2,689.25
R2: 0.952
Random Forest Regressor:
MSE: 4,725,140.26
RMSE: 2,173.74
R2: 0.968
XGBoost Regressor:
MSE: 5,479,236.23
RMSE: 2,340.78
R2: 0.963
City-wise Performance
Evaluation metrics (MSE, RMSE, R2) for each model in the top 10 cities were summarized.
The best-performing model for each city was identified based on RMSE.
Average Performance
Calculated average RMSE for each model across all cities.
The model with the lowest average RMSE across all cities was identified as the best overall model.
Conclusion
Based on the evaluation metrics, the Random Forest Regressor achieved the best overall performance with the lowest RMSE across the entire dataset. The city-wise analysis provided insights into how each model performs in different cities, revealing the best-performing model for each city.
Future Work
Explore additional feature engineering and data augmentation techniques.
Experiment with hyperparameter tuning for improved model performance.
Consider using advanced models or ensemble methods to further enhance predictions.
Requirements
Ensure you have the following Python packages installed:
pandas
scikit-learn
seaborn
matplotlib
xgboost
You can install the required packages using pip:

pip install pandas scikit-learn seaborn matplotlib xgboost

Usage
To run the analysis, execute the provided script. Ensure the dataset file (1651277648862_healthinsurance.csv) is in the same directory as the script.

