import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

class AutoPreprocessor:
    def __init__(self, target_col, missing_threshold=0.5, high_cardinality_threshold=15, corr_threshold=0.85):
        self.target_col = target_col
        self.missing_threshold = missing_threshold
        self.high_cardinality_threshold = high_cardinality_threshold
        self.corr_threshold = corr_threshold
        self.encoders = {}
        self.imputers = {}
        self.outlier_bounds = {}
        self.target_encoder = None
        self.cols_to_drop_missing = []
        self.cols_to_drop_cardinality = []
        self.cols_to_drop_correlation = []
        self.feature_columns_ = []
        self.is_fitted = False

    def _clean_column_names(self, df):
        df_clean = df.copy()
        df_clean.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df_clean.columns]
        clean_target = re.sub(r'[^A-Za-z0-9_]+', '_', self.target_col)
        self.target_col = clean_target
        return df_clean

    def fit(self, df):
        df_clean = self._clean_column_names(df)
        
        missing_frac = df_clean.isnull().mean()
        self.cols_to_drop_missing = missing_frac[missing_frac > self.missing_threshold].index.tolist()
        
        if self.target_col in self.cols_to_drop_missing:
            self.cols_to_drop_missing.remove(self.target_col)
            
        df_clean = df_clean.drop(columns=self.cols_to_drop_missing)
        df_clean = df_clean.dropna(subset=[self.target_col])

        y_temp = df_clean[self.target_col]
        # Fixed deprecated pandas categorical check
        if y_temp.dtype == 'object' or isinstance(y_temp.dtype, pd.CategoricalDtype):
            self.target_encoder = LabelEncoder()
            self.target_encoder.fit(y_temp)
            # Make target encoding auditable
            mapping = dict(zip(self.target_encoder.classes_, self.target_encoder.transform(self.target_encoder.classes_)))
            print(f"[PREPROCESS] Target encoding map: {mapping}")

        X = df_clean.drop(columns=[self.target_col])
        X_numeric = pd.DataFrame(index=X.index) # Strictly for correlation math
        candidate_cols = []

        for col in X.columns:
            dtype = X[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.outlier_bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

                imputer = SimpleImputer(strategy='median')
                X_numeric[col] = imputer.fit_transform(X[[col]]).ravel()
                self.imputers[col] = imputer
                candidate_cols.append(col)
                
            else:
                if X[col].nunique() > self.high_cardinality_threshold:
                    self.cols_to_drop_cardinality.append(col)
                    continue
                    
                imputer = SimpleImputer(strategy='most_frequent')
                filled = imputer.fit_transform(X[[col]])
                self.imputers[col] = imputer

                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                encoder.fit(filled)
                self.encoders[col] = encoder
                candidate_cols.append(col)

        # Apply Correlation Filter ONLY on genuine numeric columns
        if not X_numeric.empty:
            corr_matrix = X_numeric.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            self.cols_to_drop_correlation = [column for column in upper_tri.columns if any(upper_tri[column] > self.corr_threshold)]
        
        self.feature_columns_ = [c for c in candidate_cols if c not in self.cols_to_drop_correlation]
        self.is_fitted = True
        return self

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .transform().")

        df_clean = self._clean_column_names(df)
        drop_now = [c for c in self.cols_to_drop_missing + self.cols_to_drop_cardinality + self.cols_to_drop_correlation if c in df_clean.columns]
        df_clean = df_clean.drop(columns=drop_now, errors='ignore')

        has_target = self.target_col in df_clean.columns
        y = None
        
        if has_target:
            df_clean = df_clean.dropna(subset=[self.target_col])
            y = df_clean[self.target_col]
            if self.target_encoder is not None:
                y = pd.Series(self.target_encoder.transform(y), index=y.index)

        X = pd.DataFrame(index=df_clean.index)
        for col in self.feature_columns_:
            if col not in df_clean.columns:
                df_clean[col] = np.nan
                
            if col in self.outlier_bounds:
                lower, upper = self.outlier_bounds[col]
                df_clean[col] = np.clip(df_clean[col], lower, upper)
                
            filled = self.imputers[col].transform(df_clean[[col]])
            X[col] = self.encoders[col].transform(filled).ravel() if col in self.encoders else filled.ravel()

        return (X, y) if has_target else X

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)