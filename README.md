
# Health Insurance Claim Prediction

Overview:

This project aims to predict health insurance claims using a dataset containing various features related to individuals' health and demographics. Multiple machine learning models are used to predict the claim amount based on these features, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, and XGBoost Regressor. The models are evaluated on their performance across different cities to identify the best-performing model and to understand which model works best in different contexts.

Data:

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


## Methodology:



1. Data Preprocessing:

   Handled missing values by filling numerical columns with the   mean and categorical columns with the mode.

   Encoded categorical variables using LabelEncoder.

2. Feature and Target Variable:

   Features (X) include all columns except claim.
 Target variable (y) is the claim column.

3. Data Splitting:
   Split the data into training and testing sets with an 80-20 split.
4. Standardization:
    Standardized features using StandardScaler to ensure that all features contribute equally to the model.
5. Model Training and Evaluation:
   
   Trained four different models:

    a. Linear Regression

    b.Decision Tree Regressor

    c.Random Forest Regressor

    d.XGBoost Regressor

  Evaluated models using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) metrics.

6. City-wise Analysis:
 Trained and evaluated models separately for each city.

 Created a summary of model performance across the top 10 cities.



## Installation

Ensure you have the following Python packages installed:
pandas
scikit-learn
seaborn
matplotlib
xgboost
You can install the required packages using pip:

```bash
  pip install pandas scikit-learn seaborn matplotlib xgboost

```
   
## Usage/Examples


To run the analysis, execute the provided script. Ensure the dataset file (1651277648862_healthinsurance.csv) is in the same directory as the script.



