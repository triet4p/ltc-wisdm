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
    Một lớp Inferencer có thể tái sử dụng để đánh giá các mô hình PyTorch.
    """
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Khởi tạo PyTorchInferencer.

        Args:
            model (nn.Module): Mô hình PyTorch đã được huấn luyện.
            device (torch.device, optional): Thiết bị để chạy inference. Tự động phát hiện nếu None.
        """
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"Inferencer đã được khởi tạo trên thiết bị: {self.device}")

    def predict(self, data_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """
        Lấy các dự đoán (logits) và nhãn thật từ DataLoader.

        Args:
            data_loader (DataLoader): DataLoader chứa dữ liệu cần dự đoán.

        Returns:
            tuple[np.ndarray, np.ndarray]: Một tuple chứa (y_preds_raw, y_true).
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
        API chính để đánh giá toàn diện mô hình.

        Args:
            data_loader (DataLoader): DataLoader chứa dữ liệu kiểm tra.
            metrics_dict (Dict[str, callable]): Một dictionary chứa tên metric và hàm tính toán tương ứng.
                                                Ví dụ: {"f1_score": f1_score}

        Returns:
            Dict[str, Any]: Một dictionary chứa kết quả của các metrics và confusion matrix.
        """
        y_preds_raw, y_true = self.predict(data_loader)
        y_pred_classes = np.argmax(y_preds_raw, axis=1)
        
        report = {}
        # Tính toán các metrics được yêu cầu
        for metric_name, metric_func in metrics_dict.items():
            try:
                # Một số hàm metric cần các tham số đặc biệt (ví dụ: average)
                if "f1" in metric_name or "precision" in metric_name or "recall" in metric_name:
                    report[metric_name] = metric_func(y_true, y_pred_classes, average='weighted')
                else:
                    report[metric_name] = metric_func(y_true, y_pred_classes)
            except Exception as e:
                print(f"Lỗi khi tính toán metric '{metric_name}': {e}")
        
        # Luôn tính toán confusion matrix
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
        """Vẽ confusion matrix một cách đẹp mắt."""
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
        plt.ylabel('Nhãn thật (True Label)')
        plt.xlabel('Nhãn dự đoán (Predicted Label)')
        plt.show()