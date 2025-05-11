import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineering:
    def __init__(self):
        self.minmax_scalers = {}
        
    def ordinalEncoding(self, df: pd.DataFrame):
        encoded_df = df.copy()
        for col in df.columns:
            unique_values = df[col].unique()
            values_dict = {val: idx for idx, val in enumerate(sorted(unique_values))}
            encoded_df[col] = encoded_df[col].map(values_dict)
        return encoded_df
    
    def categoricalEncoding(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded_df = df.copy()
        for col in encoded_df.columns:
            encoded_df[col] = encoded_df[col].astype('category').cat.codes
        return encoded_df
    
    def normalize_feature(self, df: pd.DataFrame, columns, fit :bool) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if fit:
                self.minmax_scalers[col] = MinMaxScaler().fit(df[[col]])
            if col in self.minmax_scalers:
                df[col] = self.minmax_scalers[col].transform(df[[col]]).flatten()
        return df
    
