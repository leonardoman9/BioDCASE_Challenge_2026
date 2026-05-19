from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

try:
    from ai_edge_litert.interpreter import Interpreter
except Exception:  # pragma: no cover - fallback when ai_edge_litert is unavailable
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

try:
    from biodcase_tiny.embedded.esp_monitor_parser import compute_macs
except Exception:  # pragma: no cover - submission host-side still works without it
    def compute_macs(_path):
        return None


class InferenceHandler:
    """
    Submission inference handler for the waveform model with frozen frontend.
    """

    def __init__(self, cfg: dict | None = None, **kwargs) -> None:
        self.cfg = {}
        self.kwargs = kwargs
        self.cfg_init(cfg or {})

        for python_path in self.cfg["add_python_paths"]:
            resolved = str((Path(__file__).resolve().parent / python_path).resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)

        self.model_dir = Path(self.cfg["model_dir"])
        self.label_dict = yaml.safe_load(open(self.cfg["label_dict_yaml_path"], "r"))["label_dict"]
        self.feature_handler = self._instantiate(
            self.cfg["feature_handler"]["module"],
            self.cfg["feature_handler"]["attr"],
            self.cfg["feature_handler"].get("args", []),
            self.cfg["feature_handler"].get("kwargs", {}),
        )

        self.model = None
        self.target_model_file = None
        self.target_tflite_model_file = None
        self.tflite_model_interpreter = None
        self.feature_shape = self.feature_handler.extract(
            np.zeros(self.cfg["target_sample_rate"] * self.cfg["target_wav_length_sec"], dtype=np.float32),
            fs=self.cfg["target_sample_rate"],
        ).shape
        self.macs_model = None
        self.macs_tflite = None
        self.num_params_model = None
        self.num_params_tflite = None

        self._load_model()
        self._load_tflite()

    def cfg_init(self, cfg: dict) -> None:
        defaults = {
            "add_python_paths": [],
            "target_sample_rate": 24000,
            "target_wav_length_sec": 3,
            "model_dir": "./your_submission_model",
            "saved_model_extension": ".pth",
            "tflite_model_extension": ".tflite",
            "label_dict_yaml_path": "./your_submission_model/label_dict.yaml",
            "feature_handler": {
                "module": "feature_handler",
                "attr": "FeatureHandler",
                "args": [],
                "kwargs": {
                    "target_sample_rate": 24000,
                    "target_wav_length_sec": 3,
                    "normalize_peak": True,
                    "add_batch_dimension": True,
                    "channel_last": True,
                },
            },
        }
        merged = defaults
        merged.update(cfg)
        self.cfg = merged

    def _instantiate(self, module_name: str, attr_name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        module = importlib.import_module(module_name)
        cls = getattr(module, attr_name)
        return cls(*args, **kwargs)

    def _resolve_cfg(self, artifact: dict[str, Any]) -> dict[str, Any]:
        if "cfg" in artifact and isinstance(artifact["cfg"], dict):
            return artifact["cfg"]
        hyper_parameters = artifact.get("hyper_parameters")
        if isinstance(hyper_parameters, dict):
            if "model" in hyper_parameters and "data" in hyper_parameters:
                return hyper_parameters
            cfg = hyper_parameters.get("cfg")
            if isinstance(cfg, dict):
                return cfg
        raise ValueError("Could not resolve model cfg from saved artifact")

    def _resolve_num_classes(self, state_dict: dict[str, torch.Tensor]) -> int:
        fc_weight = state_dict.get("model.fc.weight", state_dict.get("fc.weight"))
        if fc_weight is None:
            raise ValueError("Could not infer num_classes from state_dict")
        return int(fc_weight.shape[0])

    def _load_model(self) -> None:
        model_files = sorted(self.model_dir.glob(f"*{self.cfg['saved_model_extension']}"))
        assert len(model_files) == 1, f"Expected exactly one model file, found {len(model_files)} in {self.model_dir}"
        self.target_model_file = model_files[0]

        artifact = torch.load(self.target_model_file, map_location="cpu")
        cfg = self._resolve_cfg(artifact)
        state_dict = artifact["state_dict"]

        target = cfg["model"]["target"]
        params = dict(cfg["model"].get("params") or {})
        params["num_classes"] = self._resolve_num_classes(state_dict)
        module_name, class_name = target.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_cls = getattr(module, class_name)
        self.model = model_cls(**params)

        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                model_state[key.removeprefix("model.")] = value
            else:
                model_state[key] = value

        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        if missing:
            print(f"warning: missing model keys: {missing}")
        if unexpected:
            print(f"warning: unexpected model keys: {unexpected}")

        self.model.eval()
        self.num_params_model = int(sum(param.numel() for param in self.model.parameters()))
        self.macs_model = None

    def _load_tflite(self) -> None:
        tflite_files = sorted(self.model_dir.glob(f"*{self.cfg['tflite_model_extension']}"))
        if len(tflite_files) != 1:
            print(f"warning: expected exactly one tflite file, found {len(tflite_files)} in {self.model_dir}")
            return

        self.target_tflite_model_file = tflite_files[0]
        self.tflite_model_interpreter = Interpreter(model_path=str(self.target_tflite_model_file))
        self.tflite_model_interpreter.allocate_tensors()
        self.macs_tflite = compute_macs(self.target_tflite_model_file)
        self.num_params_tflite = self._count_tflite_params()

    def _count_tflite_params(self) -> int | None:
        if self.tflite_model_interpreter is None:
            return None
        try:
            total = 0
            tensor_details = self.tflite_model_interpreter.get_tensor_details()
            for op in self.tflite_model_interpreter._get_ops_details():
                if op["op_name"] not in {"CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV", "FULLY_CONNECTED"}:
                    continue
                shape = tensor_details[op["inputs"][1]]["shape"]
                total += int(np.prod(shape).item())
            return total
        except Exception:
            return None

    def infer(self, wav, fs=24000):
        features = self.feature_handler.extract(wav, fs=fs)
        y_hat_model = self._predict_pytorch(features)
        y_hat_tflite = None
        if self.tflite_model_interpreter is not None:
            y_hat_tflite = self.tflite_inference(features)
        return y_hat_model, y_hat_tflite

    def _predict_pytorch(self, features: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(features).float()
        if x.ndim != 3:
            raise ValueError(f"Expected waveform tensor [batch, time, channels], got {tuple(x.shape)}")
        if x.shape[-1] == 1:
            x = x.transpose(1, 2).contiguous()
        with torch.no_grad():
            y_hat = self.model(x)
        return y_hat.detach().cpu().numpy()

    def tflite_inference(self, features: np.ndarray) -> np.ndarray:
        input_details = self.tflite_model_interpreter.get_input_details()
        output_details = self.tflite_model_interpreter.get_output_details()

        input_dtype = input_details[0]["dtype"]
        output_dtype = output_details[0]["dtype"]
        quantized = False

        x = features.reshape(input_details[0]["shape"])
        if input_dtype == np.int8 and output_dtype == np.int8:
            input_scale = input_details[0]["quantization_parameters"]["scales"][0]
            input_zero_point = input_details[0]["quantization_parameters"]["zero_points"][0]
            output_scale = output_details[0]["quantization_parameters"]["scales"][0]
            output_zero_point = output_details[0]["quantization_parameters"]["zero_points"][0]
            x = np.clip(x / input_scale + input_zero_point, -128, 127).astype(np.int8)
            quantized = True
        elif input_dtype == np.float32 and output_dtype == np.float32:
            x = x.astype(np.float32, copy=False)
        else:
            raise NotImplementedError(
                f"Unsupported TFLite IO dtypes: input={input_dtype}, output={output_dtype}"
            )

        self.tflite_model_interpreter.set_tensor(input_details[0]["index"], x)
        self.tflite_model_interpreter.invoke()
        y_hat = self.tflite_model_interpreter.get_tensor(output_details[0]["index"])

        if quantized:
            y_hat = (y_hat.astype(np.float32) - output_zero_point) * output_scale

        return np.asarray(y_hat)

    def info(self):
        print("\n--\nInference handler info:")
        print(f"model file:   [{self.target_model_file}]")
        print(f"tflite file:  [{self.target_tflite_model_file}]")
        print(f"label dict:   {self.label_dict}")
        print(f"feature shape:{self.feature_shape}")
        print("--\n")

    def get_label_dict(self):
        return self.label_dict

    def get_feature_shape(self):
        return self.feature_shape

    def get_model_size(self):
        return self.target_model_file.stat().st_size

    def get_model_file(self):
        return self.target_model_file

    def get_macs_model(self):
        return self.macs_model

    def get_num_params_model(self):
        return self.num_params_model

    def get_tflite_model_file(self):
        return self.target_tflite_model_file

    def get_macs_tflite(self):
        return self.macs_tflite

    def get_num_params_tflite(self):
        return self.num_params_tflite
