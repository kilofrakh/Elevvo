import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.model_selection import GridSearchCV

# Load the dataset
def load_data():
    # Download from: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
    df = pd.read_csv('loan_approval_dataset.csv')
    print("Initial dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    return df

# Data preprocessing
def preprocess_data(df):
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Drop irrelevant columns
    df = df.drop(['loan_id'], axis=1)
    
    # Convert categorical columns to numeric
    cat_cols = ['education', 'self_employed', 'loan_status']
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    # Handle missing values
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    # Feature engineering
    df['loan_to_income'] = df['loan_amount'] / (df['income_annum'] + 1)
    df['loan_term_years'] = df['loan_term'] / 12
    
    # Check class distribution
    print("\nClass distribution:")
    print(df['loan_status'].value_counts(normalize=True))
    
    return df

# Model training and evaluation
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        # Without SMOTE
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # With SMOTE
        smote_pipeline = imbpipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        smote_pipeline.fit(X_train, y_train)
        y_pred_smote = smote_pipeline.predict(X_test)
        y_prob_smote = smote_pipeline.predict_proba(X_test)[:, 1]
        
        # Store results
        results[name] = {
            'without_smote': {
                'report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob)
            },
            'with_smote': {
                'report': classification_report(y_test, y_pred_smote, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred_smote),
                'roc_auc': roc_auc_score(y_test, y_prob_smote)
            }
        }
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fpr_smote, tpr_smote, _ = roc_curve(y_test, y_prob_smote)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["without_smote"]["roc_auc"]:.2f})')
        plt.plot(fpr_smote, tpr_smote, label=f'{name} with SMOTE (AUC = {results[name]["with_smote"]["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        plt.show()
    
    return results

# Main function
def main():
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Prepare features and target
    X = df.drop(['loan_status'], axis=1)
    y = df['loan_status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train and evaluate models
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Print results
    for model_name, model_results in results.items():
        print(f"\n{'-'*50}")
        print(f"Model: {model_name}")
        print("\nWithout SMOTE:")
        print(f"Classification Report:\n{pd.DataFrame(model_results['without_smote']['report'])}")
        print(f"Confusion Matrix:\n{model_results['without_smote']['confusion_matrix']}")
        print(f"ROC AUC: {model_results['without_smote']['roc_auc']:.3f}")
        
        print("\nWith SMOTE:")
        print(f"Classification Report:\n{pd.DataFrame(model_results['with_smote']['report'])}")
        print(f"Confusion Matrix:\n{model_results['with_smote']['confusion_matrix']}")
        print(f"ROC AUC: {model_results['with_smote']['roc_auc']:.3f}")
    
    # Feature importance for tree-based models
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'Feature': df.drop(['loan_status'], axis=1).columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance - Random Forest')
    plt.show()

if __name__ == "__main__":
    main()