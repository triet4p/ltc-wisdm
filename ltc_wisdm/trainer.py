import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import shutil
from typing import Dict, Any, List, Literal, Optional, Type

class PyTorchTrainer:
    """
    A reusable Trainer class for PyTorch models.
    
    Includes training loop, validation, LR scheduling, gradient clipping,
    and checkpoint management (save best & last, load).
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler_cls: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        device: torch.device = None,
        gradient_clip_val: float = None
    ):
        """
        Initialize the PyTorchTrainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            criterion (nn.Module): Loss function for training.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            lr_scheduler_cls (Optional[Type[torch.optim.lr_scheduler._LRScheduler]], optional): Learning rate scheduler class. Defaults to None.
            lr_scheduler_kwargs (Optional[Dict[str, Any]], optional): Arguments for the learning rate scheduler. Defaults to None.
            device (torch.device, optional): Device to run training on. Defaults to CUDA if available, else CPU.
            gradient_clip_val (float, optional): Gradient clipping value. Defaults to None.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gradient_clip_val = gradient_clip_val
        
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = lr_scheduler_kwargs if lr_scheduler_kwargs else {}
        self.lr_scheduler = self._create_scheduler()
        
        self.model.to(self.device)
        self.best_checkpoint_path: Optional[str] = None
        self.last_checkpoint_path: Optional[str] = None
        
        self._epoch: int = 0
        self._history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": []
        }
        
        print(f"Trainer initialized on device: {self.device}")
        
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create a new scheduler instance."""
        if self.lr_scheduler_cls:
            return self.lr_scheduler_cls(self.optimizer, **self.lr_scheduler_kwargs)
        return None

    def _reset_model_weights(self, m: nn.Module):
        """Helper function to reset the weights of a layer if possible."""
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        
    def reset(self):
        """
        Reset the trainer to initial state.
        This includes epoch counter, history, and model weights.
        """
        self._epoch = 0
        self._train_loss = None
        self._val_loss = None
        self.best_checkpoint_path = None
        self.last_checkpoint_path = None
        for k, v in self._history.items():
            self._history[k].clear()
        
        self._reset_model_weights()
        self.optimizer.state.clear()
        
        self.lr_scheduler = self._create_scheduler()
        

    def _train_one_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """
        Train one epoch of the model.

        Args:
            train_loader (DataLoader): DataLoader for training data.

        Returns:
            tuple[float, float]: Average loss and accuracy for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training Epoch", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            if self.gradient_clip_val:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.3f}")

        avg_loss = total_loss / total_samples
        avg_acc = correct_predictions / total_samples
        return avg_loss, avg_acc

    def _validate_one_epoch(self, val_loader: DataLoader) -> tuple[float, float]:
        """
        Validate one epoch of the model.

        Args:
            val_loader (DataLoader): DataLoader for validation data.

        Returns:
            tuple[float, float]: Average loss and accuracy for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / total_samples
        avg_acc = correct_predictions / total_samples
        return avg_loss, avg_acc

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

    def load_model(self, checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, 
                   lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, device: torch.device = None):
        """
        Class method to load a full checkpoint.
        Note: model, optimizer, scheduler need to be initialized before calling this function.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
            model (nn.Module): Model to load the state into.
            optimizer (Optional[torch.optim.Optimizer]): Optimizer to load the state into. Defaults to None.
            lr_scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler to load the state into. Defaults to None.
            device (torch.device): Device to load the model onto. Defaults to CUDA if available, else CPU.

        Returns:
            tuple: A tuple containing (model, optimizer, lr_scheduler, epoch, history)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if lr_scheduler and 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        epoch = checkpoint.get('epoch', 0)
        history = checkpoint.get('history', {})
        
        print(f"Model loaded from checkpoint: {checkpoint_path}")
        
        return model, optimizer, lr_scheduler, epoch, history


    def fit(self, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            num_epochs: int,
            checkpoint_dir: Optional[str] = None,
            reload_best_checkpoint: bool = True):
        """
        Main API to start the training process.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            num_epochs (int): Number of epochs to train.
            checkpoint_dir (str, optional): Directory to save checkpoints. If None, no saving. Defaults to None.
            reload_best_checkpoint (bool): If True, reload the best checkpoint after training. Defaults to True.

        Returns:
            dict: Training history containing metrics per epoch.
        """
        history = self._history.copy()
        best_val_loss = float('inf')

        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            train_loss, train_acc = self._train_one_epoch(train_loader)
            val_loss, val_acc = self._validate_one_epoch(val_loader)
            
            epoch_time = time.time() - start_time
            
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history['val_acc'].append(val_acc)
            history["lr"].append(self.optimizer.param_groups[0]['lr'])

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # --- Checkpoint Management ---
            if checkpoint_dir:
                # 1. Save last checkpoint
                self.save_checkpoint(checkpoint_dir, "last_checkpoint.pth", epoch, val_loss)
                self.last_checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
                
                # 2. Save best checkpoint
                if val_loss < best_val_loss:
                    print(f"Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving best model...")
                    best_val_loss = val_loss
                    self.save_checkpoint(checkpoint_dir, "best_checkpoint.pth", epoch, history)
                    self.best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")

        print("--- Training completed! ---")

        # --- Load best checkpoint (optional) ---
        if reload_best_checkpoint and self.best_checkpoint_path:
            print("Loading weights from best checkpoint...")
            self.load_model(self.best_checkpoint_path, self.model)
            
        self._history = history
        
        return history