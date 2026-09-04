"""Render predicted versus ground-truth landmarks for the iBUG holdout set."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

import train_ear
from evaluate_ear import heatmap_points


PARTS = (20, 15, 15, 5)


def draw_tile(image: np.ndarray, truth: np.ndarray, predicted: np.ndarray, label: str, size: int) -> np.ndarray:
    canvas = cv2.cvtColor(np.clip(image * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    scale = np.asarray([canvas.shape[1] / 100, canvas.shape[0] / 100])
    truth_px = np.rint(truth * scale).astype(np.int32)
    predicted_px = np.rint(predicted * scale).astype(np.int32)
    offset = 0
    for count in PARTS:
        cv2.polylines(canvas, [truth_px[offset : offset + count]], False, (0, 255, 0), 2)
        cv2.polylines(canvas, [predicted_px[offset : offset + count]], False, (255, 0, 255), 2)
        offset += count
    for point in truth_px:
        cv2.circle(canvas, tuple(point), 2, (0, 255, 0), -1)
    for point in predicted_px:
        cv2.circle(canvas, tuple(point), 2, (255, 0, 255), -1)
    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    result = np.zeros((size + 28, size, 3), dtype=np.uint8)
    result[:size] = canvas
    cv2.putText(result, label, (6, size + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def grid(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    rows = (len(tiles) + columns - 1) // columns
    blank = np.zeros_like(tiles[0])
    tiles = tiles + [blank] * (rows * columns - len(tiles))
    return np.vstack([np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)])


def main() -> None:
    root = Path("results/holdout")
    with (root / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = {Path(row["image"]).stem: float(row["nme"]) for row in csv.DictReader(handle)}

    stems = train_ear.paired_stems(Path("data/CollectionA/test"))
    process = train_ear.make_process_fn(368, 100, 2.5, False)
    dataset = train_ear.make_dataset(stems, 4, process, False, 42)
    model = tf.keras.models.load_model("saved_model/saved_model_openpose_ear_v1.h5", compile=False)
    model.load_weights("saved_model/ear_checkpoints/best.weights.h5")
    predictor = tf.keras.Model(model.input, model.get_layer("s6").output)

    records = []
    index = 0
    for images, targets in dataset:
        outputs = predictor({"input_layer": images}, training=False)
        outputs = tf.image.resize(outputs, [100, 100]).numpy()
        predicted, _ = heatmap_points(outputs)
        truth, _ = heatmap_points(targets.numpy())
        for item in range(images.shape[0]):
            stem = Path(stems[index]).stem
            records.append((stem, metrics[stem], images[item].numpy(), truth[item], predicted[item]))
            index += 1

    records.sort(key=lambda record: record[1])
    all_tiles = [draw_tile(image, truth, pred, f"{stem}  NME {nme:.3f}", 220) for stem, nme, image, truth, pred in records]
    cv2.imwrite(str(root / "all-105.jpg"), grid(all_tiles, 7), [cv2.IMWRITE_JPEG_QUALITY, 90])

    middle = len(records) // 2
    selected = records[:4] + records[middle - 2 : middle + 2] + records[-4:]
    representative = [draw_tile(image, truth, pred, f"{stem}  NME {nme:.3f}", 340) for stem, nme, image, truth, pred in selected]
    cv2.imwrite(str(root / "representative-12.jpg"), grid(representative, 4), [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(root / "all-105.jpg")
    print(root / "representative-12.jpg")


if __name__ == "__main__":
    main()
