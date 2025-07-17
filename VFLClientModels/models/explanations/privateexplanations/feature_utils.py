import pandas as pd

def preprocess_features(df, feature_names):
    # One-hot encode categorical columns if present
    categorical_cols = [
        'digital_engagement_level',
        'risk_assessment',
        'recommended_account_type'
    ]
    df_proc = df.copy()
    for col in categorical_cols:
        if col in df_proc.columns:
            dummies = pd.get_dummies(df_proc[col], prefix=col)
            df_proc = pd.concat([df_proc, dummies], axis=1)
            df_proc.drop(columns=[col], inplace=True)
    # Ensure all required feature columns are present
    for feat in feature_names:
        if feat not in df_proc.columns:
            df_proc[feat] = 0
    # Reorder columns to match feature_names
    df_proc = df_proc[feature_names]
    return df_proc 