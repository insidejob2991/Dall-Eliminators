import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# --- 1. Setup and Configuration ---
RANDOM_STATE = 42

# --- 2. Load Data and Engineer Features ---
print("--- Loading and Preparing Data ---")
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Error: Make sure train.csv and test.csv are in the correct directory.")
    exit()

# Define feature sets
categorical_features = ['Lifestyle Activities']

# Prepare training data
X = train_df.drop(['Recovery Index', 'Id'], axis=1).copy()
y = train_df['Recovery Index']

# Prepare test data
X_test = test_df.drop('Id', axis=1).copy()

# Apply feature engineering to both train and test sets
print("Applying feature engineering...")
for df in [X, X_test]:
    df['Therapy_x_Health'] = df['Therapy Hours'] * df['Initial Health Score']
    df['Therapy_plus_Health'] = df['Therapy Hours'] + df['Initial Health Score']
    df['Sleep_x_Health'] = df['Average Sleep Hours'] * df['Initial Health Score']
    df['FollowUp_x_Health'] = df['Follow-Up Sessions'] * df['Initial Health Score']
    df['Therapy_x_Sleep'] = df['Therapy Hours'] * df['Average Sleep Hours']
    df['Sleep_plus_Health'] = df['Average Sleep Hours'] + df['Initial Health Score']
    df['Therapy_plus_Sleep'] = df['Therapy Hours'] + df['Average Sleep Hours']
    df['Therapy_x_FollowUp'] = df['Therapy Hours'] * df['Follow-Up Sessions']
    df['Therapy_plus_FollowUp'] = df['Therapy Hours'] + df['Follow-Up Sessions']
    df['Health_plus_FollowUp'] = df['Initial Health Score'] + df['Follow-Up Sessions']
    df['Sleep_x_FollowUp'] = df['Average Sleep Hours'] * df['Follow-Up Sessions']
    df['Sleep_plus_FollowUp'] = df['Average Sleep Hours'] + df['Follow-Up Sessions']

numerical_features = [col for col in X.columns if col not in categorical_features]

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("-" * 30)

# --- 3. Lasso Regression Model Definition ---
print("--- Configuring Lasso Regression Model ---")
# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Define the full model pipeline for Lasso Regression
# We use ElasticNet with l1_ratio=1.0 and the specified alpha
pipe_lasso = Pipeline([
    ('preprocessor', preprocessor),
    ('model', ElasticNet(
        alpha=0.0015,
        l1_ratio=1.0,  # This makes it a Lasso Regression
        random_state=RANDOM_STATE,
        max_iter=2000
    ))
])

print(f"Model configured with alpha=0.0015 and l1_ratio=1.0")
print("-" * 30)

# --- 4. Evaluate Model Performance with Cross-Validation ---
print("--- Evaluating Model with 10-Fold Cross-Validation ---")
kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(pipe_lasso, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
cv_rmse_scores = -cv_scores

print(f"Cross-Validation RMSE Scores: \n{cv_rmse_scores}")
print(f"Average 10-Fold RMSE: {cv_rmse_scores.mean():.4f}")
print(f"Standard Deviation of RMSE: {cv_rmse_scores.std():.4f}")
print("-" * 30)

# --- 5. Final Model Training and Analysis ---
print("--- Training Final Model on All Data ---")
pipe_lasso.fit(X, y)
print("Model training complete.")

# Analyze feature coefficients from the final model
print("\n--- Feature Coefficients from Final Model ---")
model = pipe_lasso.named_steps['model']
ohe_feature_names = pipe_lasso.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)

coefficients = model.coef_
coef_series = pd.Series(coefficients, index=all_feature_names)
coef_series_sorted = coef_series.abs().sort_values(ascending=False)

print("Coefficients (sorted by absolute importance):")
print(coef_series[coef_series_sorted.index])
print("-" * 30)

# --- 6. Final Prediction and Submission ---
print("--- Generating Final Predictions and Submission File ---")
predictions = pipe_lasso.predict(X_test)

print("Creating submission file...")
submission_df = pd.DataFrame({'Id': test_df['Id'], 'Recovery Index': predictions})
submission_path = 'submission_lasso.csv'
submission_df.to_csv(submission_path, index=False)

print(f"\nSubmission file saved to: {submission_path}")
print(submission_df.head())