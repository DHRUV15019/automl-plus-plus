import os
import numpy as np
import shap
import matplotlib.pyplot as plt

class AutoExplainer:
    def __init__(self, model, X_train, task_type=None, output_dir="reports"):
        self.model = model
        self.X_train = X_train
        self.task_type = task_type
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = None
        self.sample_X = None

    def calculate_shap(self, max_samples=1000):
        print(f"[EXPLAINER] Calculating SHAP values (Max {max_samples} samples)...")
        sample_size = min(len(self.X_train), max_samples)
        self.sample_X = self.X_train.sample(n=sample_size, random_state=42)

        # check_additivity=False avoids a known LightGBM/SHAP floating-point mismatch error
        raw_values = self.explainer.shap_values(self.sample_X, check_additivity=False)

        if isinstance(raw_values, list):
            self.shap_values = raw_values
        elif isinstance(raw_values, np.ndarray) and raw_values.ndim == 3:
            self.shap_values = [raw_values[:, :, i] for i in range(raw_values.shape[2])]
        else:
            self.shap_values = raw_values

        return self.shap_values

    def _get_values_for_plot(self):
        if isinstance(self.shap_values, list):
            if self.task_type == "binary_classification":
                return self.shap_values[1]
            elif self.task_type == "multiclass_classification":
                return np.mean([np.abs(v) for v in self.shap_values], axis=0)
            else:
                return self.shap_values[-1]
        return self.shap_values

    def plot_summary(self):
        if self.shap_values is None:
            raise ValueError("Calculate SHAP values first!")

        print("[EXPLAINER] Generating SHAP summary plot...")
        values = self._get_values_for_plot()

        plt.figure()
        shap.summary_plot(values, self.sample_X, show=False)
        plt.title("Feature Importance using SHAP")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "shap_summary.png")
        plt.savefig(path)
        plt.close()
        print(f"[INFO] Summary plot saved to '{path}'")

    def plot_local_explanation(self, instance_index=0):
        """Generates a waterfall plot for a single specific prediction."""
        if self.shap_values is None:
            raise ValueError("Calculate SHAP values first!")

        print(f"[EXPLAINER] Generating local SHAP plot for instance {instance_index}...")
        values = self._get_values_for_plot()

        # Safely handle expected_value which can be a scalar or an array depending on SHAP version
        expected_val = self.explainer.expected_value
        if isinstance(expected_val, list) or isinstance(expected_val, np.ndarray):
            if self.task_type == "binary_classification":
                expected_val = expected_val[1]
            else:
                expected_val = expected_val[-1]

        # Construct modern SHAP Explanation object
        explanation = shap.Explanation(
            values=values[instance_index],
            base_values=expected_val,
            data=self.sample_X.iloc[instance_index],
            feature_names=self.sample_X.columns.tolist()
        )

        plt.figure()
        shap.plots.waterfall(explanation, show=False)
        plt.title(f"Local Explanation (Instance {instance_index})")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"shap_local_{instance_index}.png")
        plt.savefig(path)
        plt.close()
        print(f"[INFO] Local explanation plot saved to '{path}'")