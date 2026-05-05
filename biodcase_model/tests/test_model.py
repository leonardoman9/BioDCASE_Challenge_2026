import torch

from biodcase_edge.models import Improved_Phi_GRU_ATT


def test_wrennet_forward_shape():
    model = Improved_Phi_GRU_ATT(
        num_classes=11,
        spectrogram_type="combined_log_linear",
        sample_rate=24000,
        hidden_dim=64,
        n_linear_filters=64,
        n_fft=1024,
        hop_length=240,
        matchbox={"base_filters": 32, "num_layers": 3, "kernel_size": 3, "dropout": 0.15},
    )
    waveform = torch.zeros(2, 1, 72000)
    logits = model(waveform)
    assert logits.shape == (2, 11)

