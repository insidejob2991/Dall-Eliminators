# Dall-Eliminators
A machine learning project to predict the Recovery Index of patients based on their treatment and lifestyle factors. This project is a submission for the Machine Learning Course.

📈 Patient Recovery Index Prediction
A machine learning project to predict the Recovery Index of patients based on their treatment and lifestyle factors. This project uses a regularized linear model (Lasso Regression) with extensive feature engineering to achieve a robust prediction.

📝 Project Overview
The goal of this project is to predict a patient's recovery index. The final model uses Lasso Regression, which is effective for models with many features as it can perform automatic feature selection by shrinking the coefficients of less important features to zero.

Model: Lasso Regression (implemented via ElasticNet with l1_ratio=1.0 and a tuned alpha of 0.0015).

Feature Engineering: Interaction features were created by multiplying and adding the original numerical features to capture more complex relationships.

Preprocessing: Numerical features are scaled using StandardScaler, and categorical features are handled with OneHotEncoder.

Evaluation: The model's performance was evaluated using 10-Fold Cross-Validation, measuring the Root Mean Squared Error (RMSE).
