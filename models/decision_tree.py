import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

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
    # Add any other interaction terms you want to test here

numerical_features = [col for col in X.columns if col not in categorical_features]

print(f"Features (X) shape: {X.shape}")
print("-" * 30)

# --- 3. Decision Tree Tuning with GridSearchCV ---
print("--- Starting Decision Tree Tuning ---")
# Define the preprocessing pipeline (only for one-hot encoding, as trees don't need scaling)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (the numerical ones)
)

# Define the full model pipeline
pipe_dt = Pipeline([
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(random_state=RANDOM_STATE))
])

# Define the hyperparameter grid
param_grid = {
    'model__max_depth': [5, 7, 10, None],
    'model__min_samples_leaf': [10, 20, 50],
    'model__min_samples_split': [20, 40, 100]
}

# Set up and run GridSearchCV
search_rmse = GridSearchCV(
    pipe_dt,
    param_grid,
    cv=KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE),
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

search_rmse.fit(X, y)

# --- Print Tuning Results ---
print("\n--- Decision Tree Results (RMSE) ---")
print(f"Best Parameters found: {search_rmse.best_params_}")
best_rmse = -search_rmse.best_score_
print(f"Average 10-Fold RMSE (from best model): {best_rmse:.4f}")
print("-" * 30)

# --- 4. Final Model Training and Analysis ---
print("--- Training Final Model on All Data ---")
# The grid search object automatically refits the best model on all data
final_model = search_rmse.best_estimator_

print("\n--- Feature Importances from Final Model ---")
# Get the model and preprocessor steps from the final pipeline
model = final_model.named_steps['model']
preprocessor_fitted = final_model.named_steps['preprocessor']

# Get the feature names after one-hot encoding
ohe_feature_names = preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(categorical_features)
# Combine with the numerical features that were passed through
all_feature_names = list(ohe_feature_names) + numerical_features

importances = model.feature_importances_
importance_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

print("Feature Importances:")
print(importance_series)
print("-" * 30)

# --- 5. Final Prediction and Submission ---
print("--- Generating Final Predictions and Submission File ---")
predictions = final_model.predict(X_test)

submission_df = pd.DataFrame({'Id': test_df['Id'], 'Recovery Index': predictions})
submission_path = 'submission_decision_tree.csv'
submission_df.to_csv(submission_path, index=False)

print(f"Submission file '{submission_path}' successfully created!")
print(submission_df.head())