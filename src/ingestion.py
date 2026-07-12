import pandas as pd
import numpy as np

class DataIngestor:
    def __init__(self, file_path, target_col):
        self.file_path = file_path
        self.target_col = target_col
        self.df = None
        self.task_type = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"[INFO] Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file at: {self.file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"File is empty: {self.file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Could not parse CSV: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error loading file: {e}")

    def detect_task_type(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")

        target_series = self.df[self.target_col]
        unique_vals = target_series.nunique()
        dtype = target_series.dtype

        if unique_vals == 2:
            self.task_type = "binary_classification"
        elif pd.api.types.is_numeric_dtype(dtype):
            # Check if it actually contains floats, not just integers labeled as numeric
            is_float_like = not np.array_equal(target_series.dropna(), target_series.dropna().astype(int))
            if unique_vals > 20 or is_float_like:
                self.task_type = "regression"
            else:
                self.task_type = "multiclass_classification"
        elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            if unique_vals > 50:
                print("[WARNING] Target column has high cardinality object type. Might be ID or Text.")
            self.task_type = "multiclass_classification"
        else:
            self.task_type = "unknown"
        
        print(f"[INFO] Detected Task Type: {self.task_type} (Unique Values: {unique_vals})")
        return self.task_type

    def get_missing_summary(self):
        if self.df is None:
            return None
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        summary = pd.DataFrame({'Missing_Count': missing, 'Missing_Percent': missing_percent})
        return summary[summary['Missing_Count'] > 0].sort_values(by='Missing_Percent', ascending=False)