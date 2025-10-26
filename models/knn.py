import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

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

# --- 3. KNN Tuning with GridSearchCV ---
print("--- Starting K-Nearest Neighbors (KNN) Tuning ---")
# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Define the full model pipeline
pipe_knn = Pipeline([
    ('preprocessor', preprocessor),
    ('model', KNeighborsRegressor(n_jobs=-1))
])

# Define the hyperparameter grid
param_grid = {
    'model__n_neighbors': [5, 7, 9, 11, 15, 19, 21],
    'model__weights': ['uniform', 'distance'],
    'model__metric': ['euclidean', 'manhattan']
}

# Set up and run GridSearchCV
search_rmse = GridSearchCV(
    pipe_knn,
    param_grid,
    cv=KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE),
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

search_rmse.fit(X, y)

# --- Print Tuning Results ---
print("\n--- KNN Results (RMSE) ---")
print(f"Best Parameters found: {search_rmse.best_params_}")
best_rmse = -search_rmse.best_score_
print(f"Average 10-Fold RMSE (from best model): {best_rmse:.4f}")
print("-" * 30)

# --- 4. Final Model Training and Prediction ---
print("--- Generating Final Predictions and Submission File ---")
# The grid search object automatically refits the best model on all data
final_model = search_rmse.best_estimator_

predictions = final_model.predict(X_test)

submission_df = pd.DataFrame({'Id': test_df['Id'], 'Recovery Index': predictions})
submission_path = 'submission_knn.csv'
submission_df.to_csv(submission_path, index=False)

print(f"Submission file '{submission_path}' successfully created!")
print(submission_df.head())