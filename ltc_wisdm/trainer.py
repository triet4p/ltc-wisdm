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
    Một lớp Trainer có thể tái sử dụng cho các mô hình PyTorch.
    
    Bao gồm vòng lặp huấn luyện, validation, LR scheduling, gradient clipping,
    và quản lý checkpoint (lưu best & last, load).
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
        self._train_loss: float = float('inf')
        self._val_loss: float = float('inf')
        
        self.best_train_loss: float = float('inf')
        self.best_val_loss: float = float('inf')
        
        print(f"Trainer đã được khởi tạo trên thiết bị: {self.device}")
        
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Tạo một instance scheduler mới."""
        if self.lr_scheduler_cls:
            return self.lr_scheduler_cls(self.optimizer, **self.lr_scheduler_kwargs)
        return None

    def _reset_model_weights(self, m: nn.Module):
        """Hàm helper để reset trọng số của một layer nếu có thể."""
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
            
    def _update_train_losses(self):
        if self.best_train_loss is None:
            self.best_train_loss = self._train_loss
        elif (self._train_loss < self.best_train_loss and 
            self.best_train_loss is not None and
            self._train_loss is not None):
            self.best_train_loss = self._train_loss
            
    def _update_val_losses(self):
        if self.best_val_loss is None:
            self.best_val_loss = self._val_loss
        elif (self._val_loss < self.best_val_loss and 
            self.best_val_loss is not None and
            self._val_loss is not None):
            self.best_val_loss = self._val_loss
        
    def reset(self):
        self._epoch = 0
        self._train_loss = None
        self._val_loss = None
        self.best_checkpoint_path = None
        self.last_checkpoint_path = None
        self.best_train_loss = None
        self.best_val_loss = None
        
        self._reset_model_weights()
        self.optimizer.state.clear()
        
        self.lr_scheduler = self._create_scheduler()
        

    def _train_one_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        # ... (Nội dung hàm này không thay đổi)
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
        # ... (Nội dung hàm này không thay đổi)
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

    def save_checkpoint(self, dir_path: str, filename: str, epoch: int, val_loss: float):
        """Lưu checkpoint, bao gồm trạng thái của model, optimizer, và scheduler."""
        os.makedirs(dir_path, exist_ok=True)
        checkpoint_path = os.path.join(dir_path, filename)
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        if self.lr_scheduler:
            state['scheduler_state_dict'] = self.lr_scheduler.state_dict()
            
        torch.save(state, checkpoint_path)
        print(f"Checkpoint đã được lưu tại: {checkpoint_path}")

    @classmethod
    def load_model(cls, checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, 
                   lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, device: torch.device = None):
        """
        Class method để load một checkpoint đầy đủ.
        Lưu ý: model, optimizer, scheduler cần được khởi tạo trước khi gọi hàm này.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Không tìm thấy file checkpoint: {checkpoint_path}")
        
        device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if lr_scheduler and 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        epoch = checkpoint.get('epoch', 0)
        val_loss = checkpoint.get('val_loss', float('inf'))
        
        print(f"Model đã được load từ checkpoint: {checkpoint_path}")
        print(f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}")
        
        return model, optimizer, lr_scheduler, epoch, val_loss


    def fit(self, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            num_epochs: int,
            checkpoint_dir: Optional[str] = None,
            reload_best_checkpoint: bool = True):
        """
        API chính để bắt đầu quá trình huấn luyện.

        Args:
            train_loader (DataLoader): DataLoader cho tập huấn luyện.
            val_loader (DataLoader): DataLoader cho tập validation.
            num_epochs (int): Số lượng epoch để huấn luyện.
            checkpoint_dir (str, optional): Thư mục để lưu checkpoints. Nếu None, không lưu. Defaults to None.
            reload_best_checkpoint (bool): Nếu True, load lại checkpoint tốt nhất sau khi huấn luyện xong. Defaults to True.
        """
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": []
        }
        best_val_loss = float('inf')

        print(f"Bắt đầu huấn luyện cho {num_epochs} epochs...")
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
            history["lr"].append(self.optimizer.param_groups[0]['lr'])
            
            self._train_loss = train_loss
            self._update_train_losses()
            self._val_loss = val_loss
            self._update_val_losses()

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # --- Quản lý Checkpoint ---
            if checkpoint_dir:
                # 1. Lưu last checkpoint
                self.save_checkpoint(checkpoint_dir, "last_checkpoint.pth", epoch, val_loss)
                self.last_checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
                
                # 2. Lưu best checkpoint
                if val_loss < best_val_loss:
                    print(f"Validation loss cải thiện ({best_val_loss:.4f} --> {val_loss:.4f}). Đang lưu model tốt nhất...")
                    best_val_loss = val_loss
                    self.save_checkpoint(checkpoint_dir, "best_checkpoint.pth", epoch, val_loss)
                    self.best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")

        print("--- Hoàn thành huấn luyện! ---")

        # --- Load lại checkpoint tốt nhất (optional) ---
        if reload_best_checkpoint and self.best_checkpoint_path:
            print("Đang load lại trọng số từ checkpoint tốt nhất...")
            self.load_model(self.best_checkpoint_path, self.model)
        
        return history