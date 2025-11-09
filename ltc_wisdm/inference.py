# src/inference.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, List

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

    def evaluate(self, data_loader: DataLoader, metrics_dict: Dict[str, callable]) -> Dict[str, Any]:
        """
        Main API for comprehensive model evaluation.

        Args:
            data_loader (DataLoader): DataLoader containing test data.
            metrics_dict (Dict[str, callable]): A dictionary containing metric names and corresponding calculation functions.
                                                Example: {"f1_score": f1_score}

        Returns:
            Dict[str, Any]: A dictionary containing metric results and confusion matrix.
        """
        y_preds_raw, y_true = self.predict(data_loader)
        y_pred_classes = np.argmax(y_preds_raw, axis=1)
        
        report = {}
        # Calculate requested metrics
        for metric_name, metric_func in metrics_dict.items():
            try:
                # Some metric functions require special parameters (e.g., average)
                if "f1" in metric_name or "precision" in metric_name or "recall" in metric_name:
                    report[metric_name] = metric_func(y_true, y_pred_classes, average='weighted')
                else:
                    report[metric_name] = metric_func(y_true, y_pred_classes)
            except Exception as e:
                print(f"Error calculating metric '{metric_name}': {e}")
        
        # Always calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        report['confusion_matrix'] = cm
        report['accuracy'] = accuracy_score(y_true, y_pred_classes)
        report['f1_score'] = {k: f1_score(y_true, y_pred_classes, average=k)
                              for k in ['micro', 'macro', 'weighted']}
        report['roc'] = {k: roc_auc_score(y_true, y_pred_classes, average=k)
                         for k in ['micro', 'macro', 'weighted']}
        
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