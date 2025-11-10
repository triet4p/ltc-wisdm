# src/inference.py

from sklearn.calibration import label_binarize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any, List

class PyTorchInferencer:
    """
    A reusable Inferencer class for evaluating PyTorch models.
    """
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Initialize PyTorchInferencer.

        Args:
            model (nn.Module): Trained PyTorch model.
            device (torch.device, optional): Device to run inference on. Automatically detected if None.
        """
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"Inferencer initialized on device: {self.device}")

    def predict(self, data_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """
        Get predictions (logits) and true labels from DataLoader.

        Args:
            data_loader (DataLoader): DataLoader containing data to predict.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing (y_preds_raw, y_true).
        """
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        return np.concatenate(all_preds), np.concatenate(all_labels)

    def evaluate(
        self, 
        data_loader: DataLoader, 
        label_metrics_dict: Dict[str, Callable] = None,
        prob_metrics_dict: Dict[str, Callable] = None
    ) -> Dict[str, Any]:
        """
        Main API for comprehensive model evaluation.

        Args:
            data_loader (DataLoader): DataLoader containing the evaluation data.
            label_metrics_dict (Dict[str, Callable], optional): 
                Dictionary of metrics that operate on predicted class labels (e.g., accuracy, f1_score).
                The functions should accept (y_true, y_pred).
            prob_metrics_dict (Dict[str, Callable], optional): 
                Dictionary of metrics that operate on class probabilities (e.g., roc_auc_score).
                The functions should accept (y_true_binarized, y_pred_proba).

        Returns:
            Dict[str, Any]: A dictionary containing metric results and the confusion matrix.
        """
        label_metrics_dict = label_metrics_dict or {}
        prob_metrics_dict = prob_metrics_dict or {}
        
        y_preds_raw, y_true = self.predict(data_loader)
        y_pred_classes = np.argmax(y_preds_raw, axis=1)
        
        report = {}
        
        # --- Calculate label-based metrics ---
        for metric_name, metric_func in label_metrics_dict.items():
            try:
                report[metric_name] = metric_func(y_true, y_pred_classes)
            except Exception as e:
                print(f"Error calculating label metric '{metric_name}': {e}")
        
        # --- Calculate probability-based metrics ---
        if prob_metrics_dict:
            # This part only runs if needed
            num_classes = y_preds_raw.shape[1]
            y_true_binarized = label_binarize(y_true, classes=range(num_classes))
            y_pred_proba = torch.softmax(torch.tensor(y_preds_raw), dim=1).numpy()
            
            for metric_name, metric_func in prob_metrics_dict.items():
                try:
                    # For multi-class ROC AUC, ensure y_true_binarized has same shape as y_pred_proba
                    # This handles the binary case where label_binarize might return a 1D array
                    if y_true_binarized.shape[1] == 1 and num_classes > 2:
                         y_true_binarized = label_binarize(y_true, classes=range(num_classes))

                    report[metric_name] = metric_func(y_true_binarized, y_pred_proba)
                except Exception as e:
                    print(f"Error calculating probability metric '{metric_name}': {e}")
        
        # Always calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        report['confusion_matrix'] = cm
        
        return report

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str = 'Confusion Matrix',
        figsize: tuple = (10, 8)
    ):
        """Plot confusion matrix in a nice way."""
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()