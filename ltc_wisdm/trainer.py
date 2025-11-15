import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
from typing import Dict, Any, Optional, Type, Callable

def accuracy_metric(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculates accuracy for classification tasks given model outputs and true labels.
    Assumes outputs are raw logits from the model.
    """
    if outputs.size(0) == 0:
        return 0.0
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

class PyTorchTrainer:
    """
    A reusable, task-agnostic Trainer class for PyTorch models.
    
    Handles training loops, validation, LR scheduling, gradient clipping,
    and checkpoint management. It is made flexible by accepting a dictionary
    of metric functions to compute during training and validation.
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler_cls: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        device: torch.device = None,
        gradient_clip_val: Optional[float] = None,
        metrics: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the PyTorchTrainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            lr_scheduler_cls (Optional): Learning rate scheduler class.
            lr_scheduler_kwargs (Optional): Arguments for the LR scheduler.
            device (torch.device, optional): Device to run training on.
            gradient_clip_val (float, optional): Gradient clipping value.
            metrics (Dict[str, Callable], optional): Dictionary of metric functions.
                Each function should accept (outputs, labels) and return a float.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gradient_clip_val = gradient_clip_val
        self.metrics = metrics if metrics else {}
        
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = lr_scheduler_kwargs if lr_scheduler_kwargs else {}
        self.lr_scheduler = self._create_scheduler()
        
        self.model.to(self.device)
        self.best_checkpoint_path: Optional[str] = None
        self.last_checkpoint_path: Optional[str] = None
        
        self._epoch: int = 0
        self._history = { "train_loss": [], "val_loss": [], "lr": [] }
        for metric_name in self.metrics.keys():
            self._history[f"train_{metric_name}"] = []
            self._history[f"val_{metric_name}"] = []
        
        print(f"Trainer initialized on device: {self.device}")
        
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create a new scheduler instance."""
        if self.lr_scheduler_cls:
            return self.lr_scheduler_cls(self.optimizer, **self.lr_scheduler_kwargs)
        return None

    def _run_epoch(self, data_loader: DataLoader, is_training: bool) -> Dict[str, float]:
        """
        Helper function to run one epoch for either training or validation.
        This reduces code duplication between training and validation loops.
        """
        self.model.train(is_training)
        
        total_loss = 0.0
        total_samples = 0
        metric_totals = {name: 0.0 for name in self.metrics.keys()}

        desc = "Training Epoch" if is_training else "Validating"
        progress_bar = tqdm(data_loader, desc=desc, leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            with torch.set_grad_enabled(is_training):
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = self.criterion(outputs, labels)

                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.gradient_clip_val:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.optimizer.step()

            # Update loss and metrics for the batch
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            for name, func in self.metrics.items():
                metric_totals[name] += func(outputs, labels) * batch_size
            
            progress_bar.set_postfix(loss=loss.item())

        # Calculate average values for the epoch
        epoch_results = {}
        epoch_results['loss'] = total_loss / total_samples if total_samples > 0 else 0.0
        for name, total_val in metric_totals.items():
            epoch_results[name] = total_val / total_samples if total_samples > 0 else 0.0
            
        return epoch_results

    def save_checkpoint(self, dir_path: str, filename: str, epoch: int):
        """Save checkpoint, including the state of model, optimizer, and scheduler."""
        os.makedirs(dir_path, exist_ok=True)
        checkpoint_path = os.path.join(dir_path, filename)
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self._history,
        }
        if self.lr_scheduler:
            state['scheduler_state_dict'] = self.lr_scheduler.state_dict()
            
        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model state_dict from a checkpoint file."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self._epoch = checkpoint["epoch"]
        self._history = checkpoint["history"]
        print(f"Model weights loaded from checkpoint: {checkpoint_path}")


    def fit(self, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            num_epochs: int,
            checkpoint_dir: Optional[str] = None,
            reload_best_checkpoint: bool = True):
        """
        Main API to start the training process.
        """
        best_val_loss = float('inf')

        print(f"Starting training for {num_epochs} epochs...")
        last_epoch = self._epoch
        for epoch in range(last_epoch + 1, num_epochs + last_epoch + 1):
            self._epoch = epoch
            start_time = time.time()
            
            train_results = self._run_epoch(train_loader, is_training=True)
            val_results = self._run_epoch(val_loader, is_training=False)
            
            epoch_time = time.time() - start_time
            
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_results['loss'])
                else:
                    self.lr_scheduler.step()
            
            # Update history
            self._history["train_loss"].append(train_results['loss'])
            self._history["val_loss"].append(val_results['loss'])
            self._history["lr"].append(self.optimizer.param_groups[0]['lr'])
            for name in self.metrics.keys():
                self._history[f"train_{name}"].append(train_results[name])
                self._history[f"val_{name}"].append(val_results[name])

            # Log results to console
            log_str = (
                f"Epoch {epoch}/{num_epochs + last_epoch + 1} | Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_results['loss']:.4f} | Val Loss: {val_results['loss']:.4f}"
            )
            for name in self.metrics.keys():
                log_str += f" | Train {name.capitalize()}: {train_results[name]:.4f} | Val {name.capitalize()}: {val_results[name]:.4f}"
            log_str += f" | LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            print(log_str)

            # Checkpoint Management
            val_loss = val_results['loss']
            if checkpoint_dir:
                self.save_checkpoint(checkpoint_dir, "last_checkpoint.pth", epoch)
                self.last_checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
                
                if val_loss < best_val_loss:
                    print(f"Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving best model...")
                    best_val_loss = val_loss
                    self.save_checkpoint(checkpoint_dir, "best_checkpoint.pth", epoch)
                    self.best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")

        print("--- Training completed! ---")

        if reload_best_checkpoint and self.best_checkpoint_path:
            print("Loading weights from best checkpoint...")
            self.load_checkpoint(self.best_checkpoint_path)
            
        return self._history