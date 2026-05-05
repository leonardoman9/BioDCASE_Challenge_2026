from pathlib import Path

from biodcase_edge.data.dataset import BioDCASEDataset, build_class_map


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "BioDCASE2026_TinyML_Development_Dataset"


def test_class_map_has_11_classes():
    class_map = build_class_map(DATASET_DIR)
    assert len(class_map) == 11
    assert "Background" in class_map


def test_fixed_split_counts():
    class_map = build_class_map(DATASET_DIR)
    train = BioDCASEDataset(DATASET_DIR, split="train", class_map=class_map)
    val = BioDCASEDataset(DATASET_DIR, split="validation", class_map=class_map)
    assert len(train) == 2200
    assert len(val) == 549


def test_sample_shape_and_label():
    class_map = build_class_map(DATASET_DIR)
    dataset = BioDCASEDataset(DATASET_DIR, split="train", class_map=class_map)
    waveform, label = dataset[0]
    assert waveform.shape == (1, 72000)
    assert 0 <= int(label) < 11

