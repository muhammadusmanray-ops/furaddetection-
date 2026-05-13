import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_transaction_data(df):
    """
    Standardizes and cleans transaction data for model inference.
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(method='ffill')
    
    # Scaling logic
    scaler = StandardScaler()
    if 'Amount' in df.columns:
        df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
        
    return df

if __name__ == "__main__":
    print("Data Cleaner Utility Initialized.")
