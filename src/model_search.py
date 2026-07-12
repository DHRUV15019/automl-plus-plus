import optuna
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

class AutoModelSearch:
    def __init__(self, task_type, n_trials=50, timeout=600):
        self.task_type = task_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_model = None
        self.best_params = None
        self.study = None
        self.trials_df = None

    def _objective(self, trial, X, y):
        param = {
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }

        # DYNAMIC IMBALANCE HANDLING
        if self.task_type == "binary_classification":
            pos_count = np.sum(y == 1)
            neg_count = len(y) - pos_count
            imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0
            
            # Restrict Optuna to tune weight logically around the actual data ratio
            lower_bound = max(0.5, imbalance_ratio * 0.5)
            upper_bound = max(1.5, imbalance_ratio * 2.0)
            param['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', lower_bound, upper_bound)
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        elif self.task_type == "multiclass_classification":
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=3, shuffle=True, random_state=42)

        scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if "classification" in self.task_type:
                model = lgb.LGBMClassifier(**param, random_state=42)
                model.fit(X_tr, y_tr)
                
                if self.task_type == "binary_classification":
                    preds_proba = model.predict_proba(X_val)[:, 1]
                    scores.append(roc_auc_score(y_val, preds_proba))
                else:
                    preds = model.predict(X_val)
                    scores.append(accuracy_score(y_val, preds))
            else:
                model = lgb.LGBMRegressor(**param, random_state=42)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                scores.append(np.sqrt(mean_squared_error(y_val, preds)))

        return np.mean(scores)

    def optimize_and_train(self, X_train, y_train):
        print(f"\n[MODEL SEARCH] Starting Optuna tuning for {self.task_type} (Max Trials: {self.n_trials}, Timeout: {self.timeout}s)...")
        
        direction = "maximize" if "classification" in self.task_type else "minimize"
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction=direction, sampler=sampler)
        
        study.optimize(lambda trial: self._objective(trial, X_train, y_train), 
                       n_trials=self.n_trials, timeout=self.timeout)
        
        self.study = study
        self.best_params = study.best_params
        
        # Save trials dataframe as suggested by Claude for future experiment logging
        self.trials_df = study.trials_dataframe()
        
        print(f"[MODEL SEARCH] Best Params found: {self.best_params}")
        
        print("[MODEL SEARCH] Training final model with best parameters...")
        if "classification" in self.task_type:
            self.best_model = lgb.LGBMClassifier(**self.best_params, random_state=42)
        else:
            self.best_model = lgb.LGBMRegressor(**self.best_params, random_state=42)
            
        self.best_model.fit(X_train, y_train)
        return self.best_model

    def evaluate(self, X_test, y_test):
        if self.best_model is None:
            raise ValueError("Model is not trained yet!")
            
        print("\n--- FINAL EVALUATION METRICS ---")
        if self.task_type == "binary_classification":
            preds_proba = self.best_model.predict_proba(X_test)[:, 1]
            preds_class = self.best_model.predict(X_test)
            auc = roc_auc_score(y_test, preds_proba)
            acc = accuracy_score(y_test, preds_class)
            f1 = f1_score(y_test, preds_class)
            
            print(f"Test AUC:      {auc:.4f}")
            print(f"Test Accuracy: {acc:.4f}")
            print(f"Test F1-Score: {f1:.4f}")
            return auc
            
        elif self.task_type == "multiclass_classification":
            preds_class = self.best_model.predict(X_test)
            acc = accuracy_score(y_test, preds_class)
            f1_macro = f1_score(y_test, preds_class, average='macro')
            print(f"Test Accuracy: {acc:.4f}")
            print(f"Test Macro-F1: {f1_macro:.4f}")
            return acc
            
        else:
            preds = self.best_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            print(f"Test RMSE: {rmse:.4f}")
            return rmse