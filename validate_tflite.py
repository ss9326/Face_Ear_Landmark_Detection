"""Compare an exported ear TFLite model with Keras on the iBUG holdout set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

import train_ear
from evaluate_ear import heatmap_points
from export_tflite import load_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("saved_model/saved_model_openpose_ear_v1.h5"))
    parser.add_argument("--weights", type=Path, default=Path("saved_model/ear_checkpoints/best.weights.h5"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/CollectionA"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tflite_validation"))
    parser.add_argument("--input-size", type=int, default=368)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N images")
    return parser.parse_args()


def set_input(interpreter: tf.lite.Interpreter, detail: dict, value: np.ndarray) -> None:
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.integer):
        scale, zero_point = detail["quantization"]
        if scale == 0:
            raise ValueError("Quantized TFLite input has no scale")
        value = np.round(value / scale + zero_point)
        value = np.clip(value, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    else:
        value = value.astype(dtype)
    interpreter.set_tensor(detail["index"], value)


def get_output(interpreter: tf.lite.Interpreter, detail: dict) -> np.ndarray:
    value = interpreter.get_tensor(detail["index"])
    if np.issubdtype(value.dtype, np.integer):
        scale, zero_point = detail["quantization"]
        value = (value.astype(np.float32) - zero_point) * scale
    return value.astype(np.float32)


def normalized_errors(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    diagonal = float(np.linalg.norm(truth.max(axis=0) - truth.min(axis=0)))
    return np.linalg.norm(prediction - truth, axis=1) / max(diagonal, 1e-6)


def main() -> None:
    args = parse_args()
    if not args.tflite.is_file():
        raise SystemExit(f"TFLite model not found: {args.tflite}")
    stems = train_ear.paired_stems(args.data_dir / "test")
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        stems = stems[: args.limit]
    process = train_ear.make_process_fn(args.input_size, 100, 2.5, False)
    dataset = train_ear.make_dataset(stems, 1, process, False, 42)

    keras_model = load_predictor(args.model, args.weights)
    interpreter = tf.lite.Interpreter(model_path=str(args.tflite))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    expected_shape = [1, args.input_size, args.input_size, 3]
    if input_detail["shape"].tolist() != expected_shape:
        raise SystemExit(f"TFLite input is {input_detail['shape'].tolist()}, expected {expected_shape}")

    rows: list[dict[str, float | str]] = []
    for stem, (images, targets) in zip(stems, dataset):
        image = images.numpy().astype(np.float32)
        keras_heatmaps = keras_model({"input_layer": images}, training=False).numpy()
        set_input(interpreter, input_detail, image)
        interpreter.invoke()
        tflite_heatmaps = get_output(interpreter, output_detail)
        if not np.isfinite(tflite_heatmaps).all():
            bad = int(tflite_heatmaps.size - np.isfinite(tflite_heatmaps).sum())
            raise SystemExit(f"TFLite produced {bad} NaN/Inf heatmap values for {stem}.png")
        if tflite_heatmaps.shape != keras_heatmaps.shape:
            raise SystemExit(
                f"Output shape mismatch: TFLite {tflite_heatmaps.shape}, Keras {keras_heatmaps.shape}"
            )

        keras_100 = tf.image.resize(keras_heatmaps, [100, 100]).numpy()
        tflite_100 = tf.image.resize(tflite_heatmaps, [100, 100]).numpy()
        keras_points, _ = heatmap_points(keras_100)
        tflite_points, _ = heatmap_points(tflite_100)
        truth_points, _ = heatmap_points(targets.numpy())
        truth = truth_points[0]
        keras_error = normalized_errors(keras_points[0], truth)
        tflite_error = normalized_errors(tflite_points[0], truth)
        disagreement = normalized_errors(tflite_points[0], keras_points[0])
        difference = np.abs(tflite_heatmaps - keras_heatmaps)
        rows.append(
            {
                "image": f"{stem}.png",
                "keras_nme": float(keras_error.mean()),
                "tflite_nme": float(tflite_error.mean()),
                "landmark_disagreement": float(disagreement.mean()),
                "heatmap_mean_abs_diff": float(difference.mean()),
                "heatmap_max_abs_diff": float(difference.max()),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.tflite.stem}_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    summary = {
        "tflite": str(args.tflite),
        "images": len(rows),
        "keras_mean_nme": mean("keras_nme"),
        "tflite_mean_nme": mean("tflite_nme"),
        "mean_landmark_disagreement": mean("landmark_disagreement"),
        "mean_heatmap_absolute_difference": mean("heatmap_mean_abs_diff"),
        "maximum_heatmap_absolute_difference": float(max(row["heatmap_max_abs_diff"] for row in rows)),
        "preprocessing": "ground-truth-guided crop matching train_ear.py validation",
    }
    summary_path = args.output_dir / f"{args.tflite.stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Per-image metrics: {csv_path}")


if __name__ == "__main__":
    main()
