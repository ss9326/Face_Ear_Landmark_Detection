"""Evaluate the standalone ear model on every iBUG holdout image."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

import train_ear


def heatmap_points(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch, height, width, channels = heatmaps.shape
    flat = heatmaps.reshape(batch, height * width, channels)
    indices = flat.argmax(axis=1)
    confidence = flat.max(axis=1)
    x = indices % width
    y = indices // width
    return np.stack((x, y), axis=-1).astype(np.float32), confidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/CollectionA"))
    parser.add_argument("--model", type=Path, default=Path("saved_model/saved_model_openpose_ear_v1.h5"))
    parser.add_argument("--weights", type=Path, default=Path("saved_model/ear_checkpoints/best.weights.h5"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("results/holdout"))
    args = parser.parse_args()

    stems = train_ear.paired_stems(args.data_dir / "test")
    process = train_ear.make_process_fn(368, 100, 2.5, False)
    dataset = train_ear.make_dataset(stems, args.batch_size, process, False, 42)

    model = tf.keras.models.load_model(args.model, compile=False)
    model.load_weights(args.weights)
    predictor = tf.keras.Model(model.input, model.get_layer("s6").output)

    rows: list[dict[str, float | str]] = []
    stem_index = 0
    for images, targets in dataset:
        predictions = predictor({"input_layer": images}, training=False).numpy()
        predictions = tf.image.resize(predictions, [100, 100]).numpy()
        predicted_points, confidence = heatmap_points(predictions)
        target_points, _ = heatmap_points(targets.numpy())
        for batch_index in range(images.shape[0]):
            truth = target_points[batch_index]
            predicted = predicted_points[batch_index]
            diagonal = float(np.linalg.norm(truth.max(axis=0) - truth.min(axis=0)))
            errors = np.linalg.norm(predicted - truth, axis=1) / max(diagonal, 1e-6)
            rows.append(
                {
                    "image": f"{stems[stem_index]}.png",
                    "nme": float(errors.mean()),
                    "median_error": float(np.median(errors)),
                    "pck_05": float((errors <= 0.05).mean()),
                    "pck_10": float((errors <= 0.10).mean()),
                    "mean_peak": float(confidence[batch_index].mean()),
                }
            )
            stem_index += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    nme = np.asarray([row["nme"] for row in rows], dtype=np.float64)
    summary = {
        "images": len(rows),
        "landmarks_per_image": train_ear.LANDMARK_SIZE,
        "mean_nme": float(nme.mean()),
        "median_nme": float(np.median(nme)),
        "p90_nme": float(np.percentile(nme, 90)),
        "worst_nme": float(nme.max()),
        "mean_pck_05": float(np.mean([row["pck_05"] for row in rows])),
        "mean_pck_10": float(np.mean([row["pck_10"] for row in rows])),
        "best_image": min(rows, key=lambda row: row["nme"])["image"],
        "worst_image": max(rows, key=lambda row: row["nme"])["image"],
        "metric": "mean landmark distance / ground-truth ear bounding-box diagonal",
        "preprocessing": "ground-truth-guided crop matching train_ear.py validation",
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Per-image metrics: {csv_path}")


if __name__ == "__main__":
    main()
