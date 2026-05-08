from biodcase_edge.cli.common import load_config


def test_debug_config_loads():
    cfg = load_config("debug")
    assert cfg.data.sample_rate == 24000
    assert cfg.model.params.hidden_dim == 64
    assert cfg.trainer.fast_dev_run == 2


def test_checkpoint_monitors_epoch_level_metric():
    cfg = load_config("baseline_57k")
    assert cfg.checkpoint.monitor == "val_macro_f1"
    assert cfg.early_stopping.monitor == "val_macro_f1"
