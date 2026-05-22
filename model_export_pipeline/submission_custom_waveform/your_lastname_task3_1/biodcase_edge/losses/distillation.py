import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining hard and soft targets.
    
    This loss function trains a student model to both:
    1. Predict correct hard labels (standard classification)
    2. Match the soft probability distribution from a teacher model
    """
    
    def __init__(self, alpha=0.5, temperature=4.0, reduction='mean'):
        """
        Initialize distillation loss.
        
        Args:
            alpha: Weight balance between hard and soft loss (0-1)
                  - alpha=0: Only hard labels (standard training)
                  - alpha=1: Only soft labels (pure distillation)
                  - alpha=0.5: Equal weight to both
            temperature: Temperature scaling for softmax (higher = softer distributions)
            reduction: How to reduce the loss ('mean', 'sum', 'none')
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        
        # Loss functions
        self.hard_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')
        
        print(f"DistillationLoss initialized:")
        print(f"  Alpha (soft weight): {alpha}")
        print(f"  Temperature: {temperature}")
        print(f"  Hard weight: {1-alpha}")
    
    def forward(self, student_logits, hard_labels, teacher_soft_labels):
        """
        Compute distillation loss.
        
        Args:
            student_logits: Raw logits from student model [batch_size, num_classes]
            hard_labels: Ground truth class indices [batch_size]
            teacher_soft_labels: Soft probabilities from teacher [batch_size, num_classes]
            
        Returns:
            Combined loss tensor
        """
        batch_size = student_logits.size(0)
        
        # Hard loss (standard cross-entropy with ground truth)
        loss_hard = self.hard_loss(student_logits, hard_labels)
        
        # Soft loss (KL divergence with teacher's soft labels)
        # Apply temperature scaling to both student and teacher
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_soft_labels / self.temperature, dim=1)
        
        loss_soft = self.soft_loss(student_soft, teacher_soft)
        
        # Scale soft loss by temperature^2 (standard practice in distillation)
        loss_soft = loss_soft * (self.temperature ** 2)
        
        # Combine losses
        total_loss = (1.0 - self.alpha) * loss_hard + self.alpha * loss_soft
        
        return total_loss, loss_hard, loss_soft
    
    def update_alpha(self, epoch, max_epochs, schedule='constant'):
        """
        Update alpha during training (optional).
        
        Args:
            epoch: Current epoch
            max_epochs: Total epochs
            schedule: 'constant', 'linear_increase', 'cosine'
        """
        if schedule == 'constant':
            return
        elif schedule == 'linear_increase':
            # Gradually increase soft loss weight
            self.alpha = min(0.8, 0.3 + 0.5 * (epoch / max_epochs))
        elif schedule == 'cosine':
            # Cosine schedule for alpha
            import math
            self.alpha = 0.3 + 0.4 * (1 + math.cos(math.pi * epoch / max_epochs)) / 2

class AdaptiveDistillationLoss(DistillationLoss):
    """
    Adaptive distillation loss that adjusts alpha based on validation performance.
    """
    
    def __init__(self, alpha=0.5, temperature=4.0, adaptation_rate=0.1):
        super().__init__(alpha, temperature)
        self.initial_alpha = alpha
        self.adaptation_rate = adaptation_rate
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def adapt_alpha(self, val_acc, patience=3):
        """
        Adapt alpha based on validation accuracy.
        
        Args:
            val_acc: Current validation accuracy
            patience: How many epochs to wait before adapting
        """
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= patience:
                # If validation accuracy is not improving, 
                # increase focus on teacher (increase alpha)
                self.alpha = min(0.9, self.alpha + self.adaptation_rate)
                self.patience_counter = 0
                print(f"Adapted alpha to {self.alpha:.3f} due to validation plateau")

def test_distillation_loss():
    """Test function for distillation loss"""
    batch_size, num_classes = 8, 70
    
    # Create dummy data
    student_logits = torch.randn(batch_size, num_classes)
    hard_labels = torch.randint(0, num_classes, (batch_size,))
    teacher_soft_labels = torch.rand(batch_size, num_classes)
    teacher_soft_labels = F.softmax(teacher_soft_labels, dim=1)  # Normalize
    
    # Test loss function
    loss_fn = DistillationLoss(alpha=0.3, temperature=4.0)
    total_loss, hard_loss, soft_loss = loss_fn(student_logits, hard_labels, teacher_soft_labels)
    
    print(f"Total loss: {total_loss:.4f}")
    print(f"Hard loss: {hard_loss:.4f}")
    print(f"Soft loss: {soft_loss:.4f}")
    print(f"Test passed!")

if __name__ == "__main__":
    test_distillation_loss() 