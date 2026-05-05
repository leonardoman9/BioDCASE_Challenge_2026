import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List
from omegaconf import ListConfig

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance in classification tasks.
    
    Focal Loss: FL(p_t) = -α_t(1-p_t)^γ log(p_t)
    
    This loss down-weights easy examples and focuses training on hard negatives.
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
                  Can be scalar or list of per-class weights
            gamma: Focusing parameter (default: 2.0)
                  Higher gamma = more focus on hard examples
            reduction: Specifies reduction method ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss computation
        """
        super(FocalLoss, self).__init__()
        
        # Handle Hydra ListConfig - convert to regular list
        if isinstance(alpha, ListConfig):
            alpha = list(alpha)
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        print(f"FocalLoss initialized:")
        print(f"  Alpha: {alpha}")
        print(f"  Gamma: {gamma}")
        print(f"  Reduction: {reduction}")

    def forward(self, logits, targets):
        """
        Compute focal loss.
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Focal loss value
        """
        # Standard cross entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Get probabilities
        p_t = torch.exp(-ce_loss)
        
        # Handle alpha weighting
        if isinstance(self.alpha, (list, tuple, torch.Tensor)):
            # Per-class alpha weights
            if isinstance(self.alpha, (list, tuple)):
                alpha_tensor = torch.tensor(self.alpha, device=logits.device, dtype=torch.float32)
            else:
                alpha_tensor = self.alpha.to(logits.device)
            
            # Use gather to select the correct alpha for each sample in the batch
            alpha_t = alpha_tensor.gather(0, targets.long())
        else:
            # Single scalar alpha
            alpha_t = self.alpha
        
        # Compute focal weight: alpha_t * (1-p_t)^gamma
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocalDistillationLoss(nn.Module):
    """
    Focal Loss combined with Knowledge Distillation.
    
    This loss function combines:
    1. Focal Loss for hard labels (better handling of class imbalance)
    2. KL Divergence with teacher's soft labels (knowledge transfer)
    """
    
    def __init__(self, alpha=0.5, gamma=2.0, temperature=4.0, 
                 class_weights=None, reduction='mean'):
        """
        Initialize Focal Distillation Loss.
        
        Args:
            alpha: Weight balance between hard and soft loss (0-1)
            gamma: Focal loss focusing parameter (default: 2.0)
            temperature: Temperature scaling for soft loss
            class_weights: Per-class weights for focal loss (optional)
            reduction: How to reduce the loss
        """
        super(FocalDistillationLoss, self).__init__()
        self.alpha = alpha  # Balance between hard and soft loss
        self.temperature = temperature
        self.reduction = reduction
        
        # Handle Hydra ListConfig for class_weights
        if isinstance(class_weights, ListConfig):
            class_weights = list(class_weights)
        
        # Loss functions
        self.focal_loss = FocalLoss(
            alpha=class_weights if class_weights is not None else 1.0,
            gamma=gamma, 
            reduction=reduction
        )
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')
        
        print(f"FocalDistillationLoss initialized:")
        print(f"  Alpha (soft weight): {alpha}")
        print(f"  Gamma (focal): {gamma}")
        print(f"  Temperature: {temperature}")
        print(f"  Hard weight: {1-alpha}")
    
    def forward(self, student_logits, hard_labels, teacher_soft_labels):
        """
        Compute focal distillation loss.
        
        Args:
            student_logits: Raw logits from student model [batch_size, num_classes]
            hard_labels: Ground truth class indices [batch_size]
            teacher_soft_labels: Soft probabilities from teacher [batch_size, num_classes]
            
        Returns:
            tuple: (total_loss, focal_hard_loss, soft_loss)
        """
        # Hard loss using Focal Loss (better for imbalanced classes)
        loss_hard = self.focal_loss(student_logits, hard_labels)
        
        # Soft loss (KL divergence with teacher's soft labels)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_soft_labels / self.temperature, dim=1)
        
        loss_soft = self.soft_loss(student_soft, teacher_soft)
        
        # Scale soft loss by temperature^2 (standard practice)
        loss_soft = loss_soft * (self.temperature ** 2)
        
        # Combine losses
        total_loss = (1.0 - self.alpha) * loss_hard + self.alpha * loss_soft
        
        return total_loss, loss_hard, loss_soft

class AdaptiveFocalDistillationLoss(FocalDistillationLoss):
    """
    Adaptive Focal Distillation Loss that adjusts gamma based on class distribution.
    """
    
    def __init__(self, alpha=0.5, gamma=2.0, temperature=4.0, 
                 class_weights=None, adaptation_rate=0.1):
        super().__init__(alpha, gamma, temperature, class_weights)
        self.initial_gamma = gamma
        self.adaptation_rate = adaptation_rate
        self.class_counts = None
        self.total_samples = 0
    
    def update_class_statistics(self, hard_labels):
        """
        Update class statistics for adaptive gamma adjustment.
        
        Args:
            hard_labels: Batch of ground truth labels
        """
        if self.class_counts is None:
            max_class = hard_labels.max().item() + 1
            self.class_counts = torch.zeros(max_class, device=hard_labels.device)
        
        # Check if we need to expand class_counts for new classes
        max_label = hard_labels.max().item()
        if max_label >= self.class_counts.size(0):
            # Expand the tensor to accommodate new classes
            old_size = self.class_counts.size(0)
            new_size = max_label + 1
            expanded_counts = torch.zeros(new_size, device=hard_labels.device)
            expanded_counts[:old_size] = self.class_counts
            self.class_counts = expanded_counts
        
        # Update counts
        for label in hard_labels:
            if label >= 0:  # Ignore negative labels
                self.class_counts[label] += 1
                self.total_samples += 1
    
    def adapt_gamma(self):
        """
        Adapt gamma based on class imbalance.
        More imbalanced datasets get higher gamma values.
        """
        if self.class_counts is None or self.total_samples == 0:
            return
        
        # Calculate class frequencies
        class_freq = self.class_counts / self.total_samples
        
        # Calculate imbalance ratio (max_freq / min_freq)
        min_freq = class_freq[class_freq > 0].min()
        max_freq = class_freq.max()
        imbalance_ratio = (max_freq / min_freq).item()
        
        # Adapt gamma: higher imbalance -> higher gamma
        # Scale gamma between initial_gamma and initial_gamma * 3
        new_gamma = self.initial_gamma * (1 + np.log(imbalance_ratio) * self.adaptation_rate)
        new_gamma = min(new_gamma, self.initial_gamma * 3)  # Cap at 3x initial
        
        # Update focal loss gamma
        self.focal_loss.gamma = new_gamma
        
        print(f"Adapted gamma to {new_gamma:.2f} (imbalance ratio: {imbalance_ratio:.2f})")
    
    def forward(self, student_logits, hard_labels, teacher_soft_labels):
        """Forward pass with adaptive gamma update."""
        # Update class statistics
        self.update_class_statistics(hard_labels)
        
        # Periodically adapt gamma (every 100 batches)
        if self.total_samples % 100 == 0 and self.total_samples > 0:
            self.adapt_gamma()
        
        return super().forward(student_logits, hard_labels, teacher_soft_labels)

def create_focal_loss_with_class_weights(class_counts, gamma=2.0, alpha_scaling=1.0):
    """
    Helper function to create Focal Loss with automatically computed class weights.
    
    Args:
        class_counts: List or tensor of samples per class
        gamma: Focal loss gamma parameter
        alpha_scaling: Scaling factor for class weights
        
    Returns:
        FocalLoss instance with computed class weights
    """
    if isinstance(class_counts, list):
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
    
    # Compute inverse frequency weights
    total_samples = class_counts.sum()
    num_classes = len(class_counts)
    
    # Inverse frequency: give more weight to rare classes
    class_weights = total_samples / (num_classes * class_counts)
    
    # Apply scaling
    class_weights = class_weights * alpha_scaling
    
    print(f"Computed class weights: {class_weights.tolist()}")
    
    return FocalLoss(alpha=class_weights.tolist(), gamma=gamma)

def test_focal_losses():
    """Test function for focal losses"""
    batch_size, num_classes = 8, 9  # 9 classes for bird classification
    
    # Create dummy data with class imbalance
    student_logits = torch.randn(batch_size, num_classes)
    hard_labels = torch.tensor([0, 0, 0, 1, 1, 2, 8, 8])  # Imbalanced
    teacher_soft_labels = torch.rand(batch_size, num_classes)
    teacher_soft_labels = F.softmax(teacher_soft_labels, dim=1)
    
    print("Testing Focal Loss:")
    focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
    focal_loss = focal_loss_fn(student_logits, hard_labels)
    print(f"Focal loss: {focal_loss:.4f}")
    
    print("\nTesting Focal Distillation Loss:")
    focal_dist_loss_fn = FocalDistillationLoss(alpha=0.3, gamma=2.0, temperature=4.0)
    total_loss, hard_loss, soft_loss = focal_dist_loss_fn(
        student_logits, hard_labels, teacher_soft_labels
    )
    print(f"Total loss: {total_loss:.4f}")
    print(f"Hard (focal) loss: {hard_loss:.4f}")
    print(f"Soft loss: {soft_loss:.4f}")
    
    print("\nTesting Adaptive Focal Distillation Loss:")
    adaptive_loss_fn = AdaptiveFocalDistillationLoss(
        alpha=0.3, gamma=2.0, temperature=4.0
    )
    total_loss2, hard_loss2, soft_loss2 = adaptive_loss_fn(
        student_logits, hard_labels, teacher_soft_labels
    )
    print(f"Adaptive total loss: {total_loss2:.4f}")
    
    print("\nTesting class weights computation:")
    class_counts = [1000, 500, 200, 100, 50, 30, 20, 10, 5]  # Highly imbalanced
    weighted_focal = create_focal_loss_with_class_weights(class_counts, gamma=2.0)
    weighted_loss = weighted_focal(student_logits, hard_labels)
    print(f"Weighted focal loss: {weighted_loss:.4f}")
    
    print("All focal loss tests passed!")

if __name__ == "__main__":
    test_focal_losses() 