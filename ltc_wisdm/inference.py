import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any, List

class PyTorchInferencer:
    """
    A reusable, task-agnostic Inferencer for evaluating PyTorch models.
    
    It predicts raw outputs and uses a flexible dictionary of metric functions
    to evaluate performance for either classification or regression tasks.
    """
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Initialize PyTorchInferencer.

        Args:
            model (nn.Module): Trained PyTorch model.
            device (torch.device, optional): Device to run inference on.
        """
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"Inferencer initialized on device: {self.device}")

    def predict(self, data_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """
        Get raw model predictions and true labels from a DataLoader.

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
        metrics_dict: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """
        Main API for model evaluation, adaptable for any task.

        Args:
            data_loader (DataLoader): DataLoader for evaluation.
            metrics_dict (Dict[str, Callable]): Dictionary of metric functions.
                Each function should accept (y_true, y_preds_raw) and compute a score.

        Returns:
            Dict[str, Any]: A dictionary containing the results of each metric.
        """
        y_preds_raw, y_true = self.predict(data_loader)
        
        report = {}
        
        for metric_name, metric_func in metrics_dict.items():
            try:
                report[metric_name] = metric_func(y_true, y_preds_raw)
            except Exception as e:
                print(f"Error calculating metric '{metric_name}': {e}")
        
        return report

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str = 'Confusion Matrix',
        figsize: tuple = (10, 8)
    ):
        """
        Plots a confusion matrix using seaborn. 
        Suitable for classification tasks.
        """
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

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_index: int = 0,
        title: str = 'True vs. Predicted Values',
        figsize: tuple = (15, 6)
    ):
        """
        Plots true vs. predicted values for a specific feature over time.
        Suitable for regression and time-series forecasting tasks.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
            feature_index (int): The index of the feature dimension to plot.
            title (str): Title of the plot.
            figsize (tuple): Figure size.
        """
        plt.figure(figsize=figsize)
        # Ensure we are plotting 1D arrays
        true_values = y_true[:, feature_index] if y_true.ndim > 1 else y_true
        pred_values = y_pred[:, feature_index] if y_pred.ndim > 1 else y_pred
        
        plt.plot(true_values, label='True Value', color='blue', alpha=0.7)
        plt.plot(pred_values, label='Predicted Value', color='red', linestyle='--')
        plt.title(f'{title} (Feature {feature_index})')
        plt.xlabel('Time Step / Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()