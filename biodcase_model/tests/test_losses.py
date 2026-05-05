import torch

from biodcase_edge.losses import FocalDistillationLoss, FocalLoss


def test_focal_loss_smoke():
    logits = torch.randn(4, 11)
    labels = torch.tensor([0, 1, 2, 3])
    loss = FocalLoss(gamma=2.0)(logits, labels)
    assert torch.isfinite(loss)


def test_focal_distillation_loss_smoke():
    logits = torch.randn(4, 11)
    labels = torch.tensor([0, 1, 2, 3])
    soft = torch.softmax(torch.randn(4, 11), dim=1)
    total, hard, soft_loss = FocalDistillationLoss(alpha=0.25, gamma=2.0, temperature=5.0)(
        logits,
        labels,
        soft,
    )
    assert torch.isfinite(total)
    assert torch.isfinite(hard)
    assert torch.isfinite(soft_loss)

