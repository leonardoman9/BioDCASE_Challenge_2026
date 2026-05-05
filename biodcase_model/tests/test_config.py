from biodcase_edge.cli.common import load_config


def test_debug_config_loads():
    cfg = load_config("debug")
    assert cfg.data.sample_rate == 24000
    assert cfg.model.params.hidden_dim == 64
    assert cfg.trainer.fast_dev_run == 2

