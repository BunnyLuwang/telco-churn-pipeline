 
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Executes initial data scrubbing to handle structural parsing inconsistencies."""
    df = df.copy()
    
    # The IBM dataset stores 'TotalCharges' as text due to blank spaces for new customers (tenure=0)
    # We force them to numeric values and replace blanks with NaN
    if 'total_charges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['total_charges'].str.strip(), errors='coerce')
        
    return df

def get_preprocessing_pipeline(categorical_cols: list, numeric_cols: list) -> ColumnTransformer:
    """Constructs a completely isolated, leak-proof column transformation pipeline."""
    
    # 1. Numeric Transformation Sub-Pipeline: Fill missing values with median, then standardize scales
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Categorical Transformation Sub-Pipeline: Fill missing values with 'missing', then One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 3. Combine both sub-pipelines into a unified structural transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop' # Explicitly drop ID columns or untracked columns safely
    )
    
    return preprocessor

if __name__ == "__main__":
    # Test execution harness using our extraction module from Day 2
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.data.fetch_data import load_churn_data
    
    try:
        # 1. Fetch raw data from your live cloud instance
        raw_df = load_churn_data()
        
        # 2. Apply initial structural scrubbing
        cleaned_df = clean_raw_data(raw_df)
        
        # 3. Segment feature columns by data type
        num_features = ['tenure', 'monthly_charges', 'total_charges']
        cat_features = [
            'gender', 'senior_citizen', 'partner', 'dependents', 'phone_service',
            'multiple_lines', 'internet_service', 'online_security', 'online_backup',
            'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
            'contract', 'paperless_billing', 'payment_method'
        ]
        
        # 4. Initialize and run the transformation engine
        pipeline = get_preprocessing_pipeline(categorical_cols=cat_features, numeric_cols=num_features)
        transformed_matrix = pipeline.fit_transform(cleaned_df)
        
        print("\n--- Preprocessing Pipeline Telemetry ---")
        print(f"Original DataFrame shape: {cleaned_df.shape}")
        print(f"Transformed math matrix shape: {transformed_matrix.shape}")
        print("Feature transformation pipelines successfully initialized and verified!")
        
    except Exception as e:
        print(f"\nProcessing core pipeline crash encountered: {e}")
