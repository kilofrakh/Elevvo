# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           classification_report, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
import time

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset from local CSV file
file_path = 'covtype.csv'  # Update this with your file path

# Define column names
column_names = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
] + [f'Wilderness_Area_{i}' for i in range(1, 5)] + [f'Soil_Type_{i}' for i in range(1, 41)] + ['Cover_Type']

# Read the CSV file with proper numeric conversion
try:
    # First, read just the first row to check if it's a header
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    
    # Check if first line matches our column names (indicating a header row)
    has_header = any(name in first_line for name in column_names)
    
    # Read the file with appropriate parameters
    df = pd.read_csv(
        file_path,
        header=0 if has_header else None,
        names=None if has_header else column_names,
        dtype={name: 'float64' for name in column_names[:10]},
        dtype.update({name: 'int8' for name in column_names[10:54]}),
        dtype.update({'Cover_Type': 'int8'}),
        low_memory=False
    )
    
    # If we had a header, drop it from the data
    if has_header:
        df = df.iloc[1:].reset_index(drop=True)
    
    # Convert all columns to numeric (coerce errors to NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values (if any)
    df = df.dropna()

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    print("Please make sure the file exists and the path is correct.")
    exit()
except Exception as e:
    print(f"Error loading CSV file: {str(e)}")
    exit()

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types and missing values:")
print(df.info())
print("\nClass distribution:")
print(df['Cover_Type'].value_counts())

# Data Preprocessing
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Feature Engineering
# Calculate distance to hydrology as Euclidean distance
df['Distance_To_Hydrology'] = np.sqrt(
    df['Horizontal_Distance_To_Hydrology']**2 + 
    df['Vertical_Distance_To_Hydrology']**2
)

# Calculate average hillshade
df['Hillshade_Avg'] = (df['Hillshade_9am'] + df['Hillshade_Noon'] + df['Hillshade_3pm']) / 3

# Split features and target
X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features (excluding binary ones)
numerical_cols = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points', 'Distance_To_Hydrology', 'Hillshade_Avg'
]

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# [Rest of your modeling code continues here...]

# Model Training and Evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate a model and return metrics and time taken."""
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    # Generate classification report
    report = classification_report(y_test, test_preds)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    
    # Calculate feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # For models without feature_importances_, use permutation importance
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importances = result.importances_mean
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    print(f"\n{model_name} Results:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Time Taken: {time_taken:.2f} seconds")
    print("\nClassification Report:")
    print(report)
    
    return {
        'model': model,
        'model_name': model_name,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'time_taken': time_taken,
        'confusion_matrix': cm,
        'feature_importances': importances,
        'classification_report': report
    }

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
}

# Evaluate models
results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_train, y_train, X_test, y_test, name)

# Hyperparameter Tuning (for the best performing model)
# Let's tune XGBoost as it often performs better
print("\nPerforming hyperparameter tuning for XGBoost...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [6, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

# Get the best model
best_xgb = grid_search.best_estimator_
print("\nBest parameters found:")
print(grid_search.best_params_)

# Evaluate the tuned model
results['XGBoost_Tuned'] = evaluate_model(
    best_xgb, X_train, y_train, X_test, y_test, 'XGBoost (Tuned)'
)

# Visualization
def plot_confusion_matrix(cm, classes, model_name, ax=None):
    """Plot a confusion matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()

def plot_feature_importance(importances, feature_names, model_name, top_n=20, ax=None):
    """Plot feature importance."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    indices = np.argsort(importances)[-top_n:]
    ax.barh(range(top_n), importances[indices], align='center')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Features - {model_name}')
    plt.tight_layout()

# Create visualizations for each model
fig, axes = plt.subplots(2, 2, figsize=(18, 16))

# Plot confusion matrices
plot_confusion_matrix(
    results['Random Forest']['confusion_matrix'],
    classes=range(1, 8),
    model_name='Random Forest',
    ax=axes[0, 0]
)

plot_confusion_matrix(
    results['XGBoost_Tuned']['confusion_matrix'],
    classes=range(1, 8),
    model_name='XGBoost (Tuned)',
    ax=axes[0, 1]
)

# Plot feature importances
plot_feature_importance(
    results['Random Forest']['feature_importances'],
    feature_names=X.columns,
    model_name='Random Forest',
    ax=axes[1, 0]
)

plot_feature_importance(
    results['XGBoost_Tuned']['feature_importances'],
    feature_names=X.columns,
    model_name='XGBoost (Tuned)',
    ax=axes[1, 1]
)

plt.show()

# Compare model performances
comparison_df = pd.DataFrame({
    'Model': [r['model_name'] for r in results.values()],
    'Train Accuracy': [r['train_accuracy'] for r in results.values()],
    'Test Accuracy': [r['test_accuracy'] for r in results.values()],
    'Time Taken (s)': [r['time_taken'] for r in results.values()]
})

print("\nModel Comparison:")
print(comparison_df.sort_values(by='Test Accuracy', ascending=False))