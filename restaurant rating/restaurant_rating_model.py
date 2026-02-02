"""
restaurant_rating_model.py

Full pipeline to predict 'Aggregate rating' from dataset.csv:
- Loads dataset
- Cleans and imputes missing values
- Encodes categorical variables (combination of OneHot and Label Encoding)
- Splits data (train/test)
- Trains Linear Regression and RandomForestRegressor
- Evaluates models (MSE, RMSE, R2)
- Shows feature importances from Random Forest
- Saves trained Random Forest model to disk (joblib)

Usage:
    python restaurant_rating_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ---------- Config ----------
DATA_PATH = "dataset.csv"   # put dataset.csv in same directory
RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_CUISINES = 20           # keep top N cuisines, group others into 'Other'
MAX_ONEHOT_UNIQUE = 15      # if a categorical col has <= this many unique values -> OneHot
SAVE_MODEL_PATH = "rf_restaurant_model.joblib"
# ----------------------------

def load_data(path):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Put dataset.csv in the same folder as this script.")
        sys.exit(1)
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # Work on a copy
    df = df.copy()
    
    # Drop columns we don't want as raw features
    drop_cols = [
        'Restaurant ID', 'Restaurant Name', 'Address',
        'Locality', 'Locality Verbose', 'Rating color', 'Rating text'
    ]
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)
    
    # Fill or standardize some columns
    # Convert Yes/No columns to binary
    bool_cols = [c for c in df.columns if df[c].dtype == object and df[c].isin(['Yes','No']).any()]
    # But sometimes values are 'Yes'/'No' or 'Yes ' etc - normalize
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    
    # Standardize columns that look boolean
    for c in ['Has Online delivery', 'Is delivering now', 'Has Table booking', 'Switch to order menu']:
        if c in df.columns:
            df[c] = df[c].replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
    
    # Handle Cuisines: reduce dimensionality by keeping top N cuisines (by frequency)
    if 'Cuisines' in df.columns:
        df['Cuisines'] = df['Cuisines'].fillna('Unknown')
        # Take first cuisine listed if multiple (common approach)
        df['Cuisines_primary'] = df['Cuisines'].apply(lambda x: str(x).split(',')[0].strip())
        top = df['Cuisines_primary'].value_counts().nlargest(TOP_CUISINES).index
        df['Cuisines_primary'] = df['Cuisines_primary'].apply(lambda x: x if x in top else 'Other')
        df.drop(columns=['Cuisines'], inplace=True)
    else:
        df['Cuisines_primary'] = 'Unknown'
    
    # Fill numeric missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Keep target separate
    if 'Aggregate rating' in numeric_cols:
        numeric_cols.remove('Aggregate rating')
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')  # enforce numeric if messy
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Handle remaining object (categorical) columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # We'll prepare transformers: for cat columns with small cardinality, OneHot; else LabelEncode
    onehot_cols = [c for c in cat_cols if df[c].nunique() <= MAX_ONEHOT_UNIQUE]
    label_cols = [c for c in cat_cols if c not in onehot_cols]
    
    # Build ColumnTransformer pipeline manually below
    return df, numeric_cols, onehot_cols, label_cols

def build_and_train(df, numeric_cols, onehot_cols, label_cols):
    # Separate features and target
    X = df.drop(columns=['Aggregate rating'])
    y = df['Aggregate rating'].astype(float)
    
    # For label columns, apply LabelEncoder (we'll transform them into numeric columns here)
    X = X.copy()
    label_encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('Unknown')
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # For boolean-like columns that were encoded as strings earlier, ensure numeric
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].fillna('Unknown').astype(str)
    
    # ColumnTransformer: onehot for onehot_cols, passthrough for numeric and label-encoded cols
   # Build a OneHotEncoder in a way that works for multiple sklearn versions
    ohe_kwargs = {'handle_unknown': 'ignore'}
    try:
        # sklearn >= 1.2 uses sparse_output
        ohe = OneHotEncoder(sparse_output=False, **ohe_kwargs)
    except TypeError:
        # older sklearn uses sparse
        ohe = OneHotEncoder(sparse=False, **ohe_kwargs)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', ohe, onehot_cols),
        ],
        remainder='passthrough'
    )

    
    # We'll build a pipeline for scaling + RF
    # Note: after ColumnTransformer, remainder columns order is: (onehot cols transformed) + (all other columns in X order)
    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Fit model pipeline
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Random Forest Results:")
    print(f"  Test samples: {len(y_test)}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")
    
    # Also train a simple Linear Regression pipeline for comparison
    lr_pipeline = Pipeline([
        ('preproc', preprocessor),
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    print("\nLinear Regression Results:")
    print(f"  MSE: {mse_lr:.4f}")
    print(f"  R2:  {r2_lr:.4f}")
    
    return pipeline, preprocessor, X_train, X_test, y_train, y_test, y_pred, X, label_encoders

def show_feature_importance(rf_pipeline, preprocessor, X):
    # Extract the RandomForestRegressor from pipeline
    rf = rf_pipeline.named_steps['rf']
    
    # We need the transformed feature names after OneHotEncoder
    # get onehot transformer
    ohe = preprocessor.named_transformers_.get('onehot', None)
    ohe_feature_names = []
    onehot_cols = preprocessor.transformers_[0][2] if preprocessor.transformers_ else []
    if ohe is not None:
        # scikit-learn >=1.0 has get_feature_names_out
        try:
            ohe_feature_names = ohe.get_feature_names_out(onehot_cols).tolist()
        except Exception:
            # fallback
            categories = ohe.categories_
            names = []
            for col, cats in zip(onehot_cols, categories):
                names.extend([f"{col}__{str(cat)}" for cat in cats])
            ohe_feature_names = names
    
    # remainder columns (the ones that were passed through) - order is the columns in X excluding onehot_cols
    remainder_cols = [c for c in X.columns if c not in onehot_cols]
    
    feature_names = ohe_feature_names + remainder_cols
    importances = rf.feature_importances_
    
    # If lengths mismatch, try to truncate or pad (defensive)
    if len(importances) != len(feature_names):
        # Attempt to align by using the transformed shape from preprocessor
        try:
            transformed = preprocessor.transform(X.iloc[:1])
            n_feats = transformed.shape[1]
            if len(importances) == n_feats:
                # doable
                pass
            else:
                # as fallback, create generic names
                feature_names = [f"f_{i}" for i in range(len(importances))]
        except Exception:
            feature_names = [f"f_{i}" for i in range(len(importances))]
    
    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print("\nTop 20 feature importances (Random Forest):")
    print(feat_imp_df.head(20).to_string(index=False))
    
    # Plot top 20
    top_n = 20
    top = feat_imp_df.head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(top['feature'][::-1], top['importance'][::-1])
    plt.xlabel("Feature importance")
    plt.title("Top features (Random Forest)")
    plt.tight_layout()
    plt.show()
    
    return feat_imp_df

def save_model(pipeline, path):
    joblib.dump(pipeline, path)
    print(f"\nSaved Random Forest pipeline to '{path}'")

def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    
    print("Preprocessing...")
    df_proc, numeric_cols, onehot_cols, label_cols = preprocess(df)
    print(f"Numeric cols: {numeric_cols}")
    print(f"OneHot candidate cols: {onehot_cols}")
    print(f"Label-encoded cols: {label_cols}")
    
    print("Training models...")
    rf_pipeline, preprocessor, X_train, X_test, y_train, y_test, y_pred, X_full, label_encoders = build_and_train(df_proc, numeric_cols, onehot_cols, label_cols)
    
    print("Analyzing feature importance...")
    feat_imp_df = show_feature_importance(rf_pipeline, preprocessor, X_full)
    
    print("Saving model...")
    save_model(rf_pipeline, SAVE_MODEL_PATH)
    
    print("\nDone.")

if __name__ == "__main__":
    main()

