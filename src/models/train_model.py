 
import os
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Inject project root folder to link up existing extraction and feature engines
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.fetch_data import load_churn_data
from src.features.build_features import clean_raw_data, get_preprocessing_pipeline

def train_production_model():
    """Extracts data over the wire, transforms it, and trains a robust XGBoost ensemble model."""
    
    # 1. Fetch and scrub the full dataset
    raw_df = load_churn_data()
    cleaned_df = clean_raw_data(raw_df)
    
    # 2. Separate independent features (X) and dependent target label (y)
    if 'Churn' not in cleaned_df.columns:
        raise KeyError("Target feature column 'Churn' missing from dataset context.")
        
    X = cleaned_df.drop(columns=['customerID', 'Churn'])
    # Encode target text string safely: 'Yes' -> 1, 'No' -> 0
    y = cleaned_df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    
    # 3. Explicitly categorize feature columns by datatype mapping
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_features = [col for col in X.columns if col not in num_features]
    
    # 4. Partition data into training (80%) and evaluation (20%) splits
    # stratify=y balances the churn distribution perfectly between splits
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Coordinate modular feature sub-pipelines
    preprocessor = get_preprocessing_pipeline(categorical_cols=cat_features, numeric_cols=num_features)
    
    print("Transforming feature spaces into mathematical training matrices...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    
    # Calculate balance ratio to counter class imbalance issues defensively
    negative_class_count = (y_train == 0).sum()
    positive_class_count = (y_train == 1).sum()
    balance_scale = negative_class_count / positive_class_count
    
    print(f"Dataset imbalance detected. Setting XGBoost balance scale ratio to: {balance_scale:.2f}")
    
    # 6. Initialize production XGBoost ensemble classifier engine
    print("Initializing production XGBoost classifier engine...")
    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=balance_scale,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train the machine learning model core
    print("Training the ensemble machine learning engine...")
    model.fit(X_train_transformed, y_train)
    
    # 7. Evaluate model efficacy metrics
    predictions = model.predict(X_val_transformed)
    probabilities = model.predict_proba(X_val_transformed)[:, 1]
    
    # 8. Output detailed metrics report
    print("\n================ PRODUCTION MODEL PERFORMANCE ================")
    print(classification_report(y_val, predictions, target_names=['Retained (0)', 'Churned (1)']))
    
    auc_score = roc_auc_score(y_val, probabilities)
    print(f"ROC-AUC Evaluation Score: {auc_score:.4f}")
    print("==============================================================")
    
    return preprocessor, model

if __name__ == "__main__":
    try:
        train_production_model()
        print("\nModel engine pipeline successfully executed and validated!")
    except Exception as e:
        print(f"\nModel engine pipeline crash encountered: {e}")
