# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            f1_score, precision_score, recall_score,
                            roc_auc_score, accuracy_score)
from imblearn.over_sampling import SMOTE
from collections import Counter

# Set random seed for reproducibility
np.random.seed(42)

try:
    df = pd.read_csv('loan_approval_dataset.csv')
except FileNotFoundError:
    print("File not found. Please ensure the dataset is in the correct path.")
    exit()

# Data Exploration
print("\n=== Dataset Info ===")
print(df.info())

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Descriptive Statistics ===")
print(df.describe())

print("\n=== Class Distribution ===")
print(df['loan_status'].value_counts())

# Data Cleaning and Preprocessing
print("\n=== Data Cleaning ===")

# Handle missing values
print("\nMissing values before handling:")
print(df.isnull().sum())

# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns.drop('loan_status')

# Impute missing values
imputer_num = SimpleImputer(strategy='median')
df[num_cols] = imputer_num.fit_transform(df[num_cols])

imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

print("\nMissing values after handling:")
print(df.isnull().sum())

# Encode categorical features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target variable
le_target = LabelEncoder()
df['loan_status'] = le_target.fit_transform(df['loan_status'])

# Feature scaling
scaler = StandardScaler()
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Handle Class Imbalance with SMOTE
print("\n=== Handling Class Imbalance ===")
print("Original class distribution:", Counter(y_train))

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:", Counter(y_train_smote))

# Model Evaluation Function
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name=""):
    print(f"\n=== {model_name} Evaluation ===")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Evaluate performance
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    return metrics

# Initialize models
log_reg = LogisticRegression(random_state=42, max_iter=1000)
dec_tree = DecisionTreeClassifier(random_state=42)

# Evaluate models on original imbalanced data
print("\n=== Models on Original Imbalanced Data ===")
metrics_logreg = evaluate_model(log_reg, X_train, y_train, X_test, y_test, "Logistic Regression")
metrics_tree = evaluate_model(dec_tree, X_train, y_train, X_test, y_test, "Decision Tree")

# Evaluate models on SMOTE-balanced data
print("\n=== Models on SMOTE-Balanced Data ===")
metrics_logreg_smote = evaluate_model(log_reg, X_train_smote, y_train_smote, X_test, y_test, "Logistic Regression with SMOTE")
metrics_tree_smote = evaluate_model(dec_tree, X_train_smote, y_train_smote, X_test, y_test, "Decision Tree with SMOTE")

# Model Comparison
print("\n=== Model Performance Comparison ===")
results = pd.DataFrame({
    'Logistic Regression (Original)': metrics_logreg,
    'Decision Tree (Original)': metrics_tree,
    'Logistic Regression (SMOTE)': metrics_logreg_smote,
    'Decision Tree (SMOTE)': metrics_tree_smote
})

print(results.transpose())

# Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")

# Fit the best model on all SMOTE data
best_model = DecisionTreeClassifier(random_state=42)
best_model.fit(X_train_smote, y_train_smote)

# Get feature importances
importances = best_model.feature_importances_
features = df.drop('loan_status', axis=1).columns

# Create DataFrame and sort
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
plt.xlabel('Importance Score')
plt.title('Top 15 Important Features for Loan Approval Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Hyperparameter Tuning for the Best Model
print("\n=== Hyperparameter Tuning ===")

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train_smote, y_train_smote)

# Get the best model
best_dt = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

# Evaluate the tuned model
print("\n=== Tuned Decision Tree Performance ===")
metrics_best_dt = evaluate_model(best_dt, X_train_smote, y_train_smote, X_test, y_test, "Tuned Decision Tree")

# Fiinal Model Comparison
print("\n=== Final Model Comparison ===")
final_results = pd.DataFrame({
    'Original Decision Tree': metrics_tree_smote,
    'Tuned Decision Tree': metrics_best_dt
})

print(final_results.transpose())

import joblib
joblib.dump(best_dt, 'best_loan_approval_model.pkl')
print("\nBest model saved as 'best_loan_approval_model.pkl'")