import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Executes initial data scrubbing to handle structural parsing inconsistencies."""
    df = df.copy()
    
    # Force TotalCharges to numeric values and replace blanks with NaN safely
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
        
    return df

def get_preprocessing_pipeline(categorical_cols: list, numeric_cols: list) -> ColumnTransformer:
    """Constructs a completely isolated, leak-proof column transformation pipeline."""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.data.fetch_data import load_churn_data
    
    try:
        raw_df = load_churn_data()
        cleaned_df = clean_raw_data(raw_df)
        
        num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        cat_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
        
        pipeline = get_preprocessing_pipeline(categorical_cols=cat_features, numeric_cols=num_features)
        transformed_matrix = pipeline.fit_transform(cleaned_df)
        
        print("\n--- Preprocessing Pipeline Telemetry ---")
        print(f"Original DataFrame shape: {cleaned_df.shape}")
        print(f"Transformed math matrix shape: {transformed_matrix.shape}")
        print("Feature transformation pipelines successfully initialized and verified!")
        
    except Exception as e:
        print(f"\nProcessing core pipeline crash encountered: {e}")
