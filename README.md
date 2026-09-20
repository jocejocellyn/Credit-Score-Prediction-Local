# Credit-Score-Prediction-Local
An end-to-end machine learning project for classifying customers into **Poor, Standard, and Good** credit score categories.  
This repository contains the main local implementation of the project, covering data preprocessing, exploratory data analysis, model development, hyperparameter tuning, evaluation, MLflow experiment tracking, and local Streamlit deployment.

## Project Workflow
Data → EDA → Preprocessing → Model Training → Hyperparameter Tuning → Evaluation → MLflow → Model Selection → Streamlit Deployment

## Dataset
The dataset contains 25.000 customer records with `Credit_Score` as the target variable.  
Identifier columns such as Unnamed, ID, Customer ID, Name, and SSN were removed because they were not required for model development.

## Data Preprocessing
The preprocessing stage included:
* Handling invalid and inconsistent values
* Identifying unrealistic numerical values
* Treating relevant outliers
* Handling missing values
* Preparing numerical and categorical features
* Splitting the dataset into training and testing sets using an 80:20 ratio
* Applying stratified splitting based on `Credit_Score`  
A reusable `ModelPreprocessor` class was implemented to organize the preprocessing workflow.

## Exploratory Data Analysis
EDA was performed to understand the target distribution and relationships between customer characteristics and credit score categories. The analysis included:
* Target distribution
* Categorical feature analysis
* Numerical feature distributions
* Correlation analysis
* Outlier analysis  
Several features showed noticeable patterns across credit score categories, including Credit Mix, Payment of Minimum Amount, Monthly Inhand Salary, Outstanding Debt, and Delay from Due Date.

## Model Development
Several machine learning models were explored, including:
* Logistic Regression
* Random Forest
* Extra Trees
* Gradient Boosting
* XGBoost  
Random Forest, Extra Trees, and XGBoost were further optimized and evaluated during the main model comparison.

## Hyperparameter Tuning
Hyperparameter tuning was performed using `RandomizedSearchCV` with:
* 20 parameter combinations
* 3-fold cross-validation
* `f1_macro` as the scoring metric  
Models were evaluated using Accuracy, Precision, Recall, and F1-Score.

## MLflow Experiment Tracking
MLflow was used to track model experiments, including:
* Hyperparameters
* Evaluation metrics
* Model artifacts
* Trained model files  
This allowed different model configurations to be compared systematically.

## Deployment
The selected model was deployed locally through a Streamlit application. The application allows users to enter customer information and obtain a predicted credit score category. Test cases were also used to verify predictions across the three target classes.  
Access the Streamlit app here: https://credit-score-prediction-jocellyn-jonathan.streamlit.app/ 
