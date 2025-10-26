import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sample_submission_df = pd.read_csv('sample_submission.csv')
except FileNotFoundError as e:
    print("Please make sure train.csv, test.csv, and sample_submission.csv are in the correct directory.")
    exit()

print("Data loaded successfully!")
print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)

X = train_df.drop(['Recovery Index', 'Id'], axis=1)
y = train_df['Recovery Index']
X_test = test_df.drop('Id', axis=1)

# --- Feature Engineering ---
# Create new features by combining existing ones

# Original features
X['Therapy_x_Health'] = X['Therapy Hours'] * X['Initial Health Score']
X['Therapy_plus_Health'] = X['Therapy Hours'] + X['Initial Health Score']
X['Sleep_x_Health'] = X['Average Sleep Hours'] * X['Initial Health Score']
X['FollowUp_x_Health'] = X['Follow-Up Sessions'] * X['Initial Health Score']
X['Therapy_x_Sleep'] = X['Therapy Hours'] * X['Average Sleep Hours']
X['Sleep_plus_Health'] = X['Average Sleep Hours'] + X['Initial Health Score']

# Added more features
X['Therapy_plus_Sleep'] = X['Therapy Hours'] + X['Average Sleep Hours']
X['Therapy_x_FollowUp'] = X['Therapy Hours'] * X['Follow-Up Sessions']
X['Therapy_plus_FollowUp'] = X['Therapy Hours'] + X['Follow-Up Sessions']
X['Health_plus_FollowUp'] = X['Initial Health Score'] + X['Follow-Up Sessions']
X['Sleep_x_FollowUp'] = X['Average Sleep Hours'] * X['Follow-Up Sessions']
X['Sleep_plus_FollowUp'] = X['Average Sleep Hours'] + X['Follow-Up Sessions']


# --- Apply to Test Set ---
X_test['Therapy_x_Health'] = X_test['Therapy Hours'] * X_test['Initial Health Score']
X_test['Therapy_plus_Health'] = X_test['Therapy Hours'] + X_test['Initial Health Score']
X_test['Sleep_x_Health'] = X_test['Average Sleep Hours'] * X_test['Initial Health Score']
X_test['FollowUp_x_Health'] = X_test['Follow-Up Sessions'] * X_test['Initial Health Score']
X_test['Therapy_x_Sleep'] = X_test['Therapy Hours'] * X_test['Average Sleep Hours']
X_test['Sleep_plus_Health'] = X_test['Average Sleep Hours'] + X_test['Initial Health Score']

# Added more features to test set
X_test['Therapy_plus_Sleep'] = X_test['Therapy Hours'] + X_test['Average Sleep Hours']
X_test['Therapy_x_FollowUp'] = X_test['Therapy Hours'] * X_test['Follow-Up Sessions']
X_test['Therapy_plus_FollowUp'] = X_test['Therapy Hours'] + X_test['Follow-Up Sessions']
X_test['Health_plus_FollowUp'] = X_test['Initial Health Score'] + X_test['Follow-Up Sessions']
X_test['Sleep_x_FollowUp'] = X_test['Average Sleep Hours'] * X_test['Follow-Up Sessions']
X_test['Sleep_plus_FollowUp'] = X_test['Average Sleep Hours'] + X_test['Follow-Up Sessions']
# --- End Feature Engineering ---

categorical_features = ['Lifestyle Activities']
numerical_features = [
    'Therapy Hours', 
    'Initial Health Score', 
    'Average Sleep Hours', 
    'Follow-Up Sessions',
    'Therapy_x_Health',      
    'Therapy_plus_Health',   
    'Sleep_x_Health',        
    'FollowUp_x_Health',     
    'Therapy_x_Sleep',       
    'Sleep_plus_Health',
    'Therapy_plus_Sleep',    # Add new feature
    'Therapy_x_FollowUp',    # Add new feature
    'Therapy_plus_FollowUp', # Add new feature
    'Health_plus_FollowUp',  # Add new feature
    'Sleep_x_FollowUp',      # Add new feature
    'Sleep_plus_FollowUp'    # Add new feature
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

print("--- Training Simple Linear Regression with K-Fold Cross-Validation ---")

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', LinearRegression())])

# --- K-Fold Cross-Validation ---
# 90:10 split means 10 splits
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation
# We use 'neg_root_mean_squared_error' as scoring
cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='neg_root_mean_squared_error')

# Scores are negative, so we multiply by -1
cv_rmse_scores = -cv_scores

print(f"K-Fold RMSE Scores: {cv_rmse_scores}")
print(f"Mean RMSE: {cv_rmse_scores.mean():.4f}")
print(f"Std Dev RMSE: {cv_rmse_scores.std():.4f}")
print("-" * 30)

# --- Final Model Training and Prediction ---
# Now, train the model on the *entire* training set for the final submission
print("Training final model on all data...")
pipeline.fit(X, y)
print("Model training complete.")
print("-" * 30)

# Calculate RMSE on the full training set (optional, just to compare with CV)
y_pred_train = pipeline.predict(X)
train_rmse = np.sqrt(mean_squared_error(y, y_pred_train))
print(f"Training RMSE on *full* dataset: {train_rmse:.4f}")

# Make predictions on the test set
predictions = pipeline.predict(X_test)

submission_df = pd.DataFrame({'Id': test_df['Id'], 'Recovery Index': predictions})

submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")

