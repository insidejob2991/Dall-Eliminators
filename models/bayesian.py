import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge

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
print("-" * 30)

# --- 3. Bayesian Ridge Regression Model ---
print("--- Configuring Bayesian Ridge Regression Model ---")
# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Define the full model pipeline
pipe_bayesian = Pipeline([
    ('preprocessor', preprocessor),
    ('model', BayesianRidge())
])

# --- 4. Evaluate Model Performance with Cross-Validation ---
print("--- Evaluating Model with 10-Fold Cross-Validation ---")
kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(
    pipe_bayesian, X, y,
    cv=kf,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
cv_rmse_scores = -cv_scores

print(f"Average 10-Fold RMSE: {cv_rmse_scores.mean():.4f}")
print(f"Standard Deviation of RMSE: {cv_rmse_scores.std():.4f}")
print("-" * 30)

# --- 5. Final Model Training and Analysis ---
print("--- Training Final Model on All Data ---")
pipe_bayesian.fit(X, y)

print("\n--- Feature Coefficients from Final Model ---")
model = pipe_bayesian.named_steps['model']
ohe_feature_names = pipe_bayesian.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)

coefficients = model.coef_
coef_series = pd.Series(coefficients, index=all_feature_names)
coef_series_sorted = coef_series.abs().sort_values(ascending=False)

print("Coefficients (sorted by absolute importance):")
print(coef_series[coef_series_sorted.index])
print("-" * 30)

# --- 6. Final Prediction and Submission ---
print("--- Generating Final Predictions and Submission File ---")
predictions = pipe_bayesian.predict(X_test)

submission_df = pd.DataFrame({'Id': test_df['Id'], 'Recovery Index': predictions})
submission_path = 'submission_bayesian_ridge.csv'
submission_df.to_csv(submission_path, index=False)

# This is the corrected line - emoji removed
print(f"Submission file '{submission_path}' successfully created!")
print(submission_df.head())