from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent / "biodcase_model"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedded_classifier_only.check_streaming_equivalence import classifier_logits_full
from embedded_classifier_only.export_stateful_streaming_onnx import BackboneStreamingWrapper
from pytorch_to_onnx import instantiate_model, load_checkpoint, resolve_cfg


DEFAULT_CHECKPOINT = PIPELINE_ROOT / "checkpoints" / "biodcase_best_06717.ckpt"
DEFAULT_FEATURES = SCRIPT_DIR / "representative_data_prenorm_full" / "validation_frontend_features.npy"
DEFAULT_ARTIFACTS = SCRIPT_DIR / "artifacts" / "stateful_streaming"
DEFAULT_BACKBONE_TFLITE = DEFAULT_ARTIFACTS / "biodcase_backbone_streaming_float32.tflite"
DEFAULT_STEP_TFLITE = DEFAULT_ARTIFACTS / "biodcase_streaming_step_float32.tflite"


def get_tflite_interpreter(model_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def run_backbone_tflite(interpreter, features: np.ndarray) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    input_shape = tuple(input_detail["shape"])
    x = features.astype(np.float32)
    if x.shape != input_shape and x.ndim == 3 and x.transpose(0, 2, 1).shape == input_shape:
        x = x.transpose(0, 2, 1)
    if x.shape != input_shape:
        raise ValueError(f"Backbone input shape mismatch: have {x.shape}, tflite expects {input_shape}")

    interpreter.set_tensor(input_detail["index"], x)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])


def detail_by_name(details: list[dict], name: str) -> dict:
    for detail in details:
        if detail["name"] == name:
            return detail
    names = [detail["name"] for detail in details]
    raise KeyError(f"Missing TFLite tensor {name!r}. Available: {names}")


def run_step_tflite_loop(interpreter, embeddings: np.ndarray) -> np.ndarray:
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    frame_detail = detail_by_name(inputs, "frame")
    hidden_detail = detail_by_name(inputs, "hidden")
    att_num_detail = detail_by_name(inputs, "att_num")
    att_den_detail = detail_by_name(inputs, "att_den")
    att_max_detail = detail_by_name(inputs, "att_max")

    hidden = np.zeros(tuple(hidden_detail["shape"]), dtype=np.float32)
    att_num = np.zeros(tuple(att_num_detail["shape"]), dtype=np.float32)
    att_den = np.zeros(tuple(att_den_detail["shape"]), dtype=np.float32)
    att_max = np.full(tuple(att_max_detail["shape"]), -1.0e9, dtype=np.float32)
    logits = None

    for step in range(embeddings.shape[1]):
        frame = embeddings[:, step, :].astype(np.float32)
        interpreter.set_tensor(frame_detail["index"], frame)
        interpreter.set_tensor(hidden_detail["index"], hidden)
        interpreter.set_tensor(att_num_detail["index"], att_num)
        interpreter.set_tensor(att_den_detail["index"], att_den)
        interpreter.set_tensor(att_max_detail["index"], att_max)
        interpreter.invoke()

        hidden = interpreter.get_tensor(outputs[0]["index"])
        att_num = interpreter.get_tensor(outputs[1]["index"])
        att_den = interpreter.get_tensor(outputs[2]["index"])
        att_max = interpreter.get_tensor(outputs[3]["index"])
        logits = interpreter.get_tensor(outputs[4]["index"])

    if logits is None:
        raise RuntimeError("No frames were processed")
    return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate stateful backbone+step TFLite chain against PyTorch")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--backbone-tflite", default=str(DEFAULT_BACKBONE_TFLITE))
    parser.add_argument("--step-tflite", default=str(DEFAULT_STEP_TFLITE))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--normalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(Path(args.checkpoint))
    cfg = resolve_cfg(checkpoint)
    model = instantiate_model(cfg, checkpoint["state_dict"]).cpu().eval()

    features_np = np.load(args.features)[: args.num_samples].astype(np.float32)
    if features_np.ndim == 4 and features_np.shape[-1] == 1:
        features_np = features_np[..., 0]
    if features_np.shape[1] != model.expected_input_features and features_np.shape[2] == model.expected_input_features:
        features_np = np.transpose(features_np, (0, 2, 1))

    backbone_tflite = get_tflite_interpreter(Path(args.backbone_tflite))
    step_tflite = get_tflite_interpreter(Path(args.step_tflite))
    backbone_torch = BackboneStreamingWrapper(model, normalize_input=args.normalize).cpu().eval()

    torch_logits = []
    tflite_logits = []
    backbone_diffs = []

    with torch.no_grad():
        torch_full = classifier_logits_full(model, torch.from_numpy(features_np), normalize=args.normalize)
        torch_embeddings = backbone_torch(torch.from_numpy(features_np)).numpy()

    for index in range(features_np.shape[0]):
        sample = features_np[index : index + 1]
        tflite_embeddings = run_backbone_tflite(backbone_tflite, sample)
        if tflite_embeddings.shape != torch_embeddings[index : index + 1].shape:
            raise ValueError(
                f"Backbone output shape mismatch: tflite {tflite_embeddings.shape}, "
                f"torch {torch_embeddings[index : index + 1].shape}"
            )
        backbone_diffs.append(np.abs(tflite_embeddings - torch_embeddings[index : index + 1]).max())
        tflite_logits.append(run_step_tflite_loop(step_tflite, tflite_embeddings)[0])
        torch_logits.append(torch_full[index].numpy())

    torch_logits_np = np.stack(torch_logits, axis=0)
    tflite_logits_np = np.stack(tflite_logits, axis=0)
    logits_diff = np.abs(torch_logits_np - tflite_logits_np)
    torch_argmax = torch_logits_np.argmax(axis=1)
    tflite_argmax = tflite_logits_np.argmax(axis=1)

    print(f"samples:               {features_np.shape[0]}")
    print(f"features:              {features_np.shape}")
    print(f"normalize:             {args.normalize}")
    print(f"backbone_max_abs_diff: {float(np.max(backbone_diffs)):.8g}")
    print(f"logits_max_abs_diff:   {float(logits_diff.max()):.8g}")
    print(f"logits_mean_abs_diff:  {float(logits_diff.mean()):.8g}")
    print(f"torch_argmax:          {torch_argmax.tolist()}")
    print(f"tflite_argmax:         {tflite_argmax.tolist()}")
    print(f"agreement:             {int((torch_argmax == tflite_argmax).sum())}/{features_np.shape[0]}")


if __name__ == "__main__":
    main()
