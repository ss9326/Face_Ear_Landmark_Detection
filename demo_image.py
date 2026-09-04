"""Run the repository's trained multi-task model on one image.

The upstream repository does not distribute the required .h5 weights.  Supply a
model produced by train_face2ear.py (or obtained from the repository author).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


EAR_PART_SIZES = (20, 15, 15, 5)
EAR_PART_COLORS = ((0, 255, 0), (255, 51, 0), (255, 204, 0), (255, 204, 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate 55 ear landmarks in a face image."
    )
    parser.add_argument("image", type=Path, help="Input face image")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("saved_model_openpose_face2ear_v1.h5"),
        help="Trained multi-task .h5 model",
    )
    parser.add_argument("--weights", type=Path, help="Optional weights checkpoint to load after the model")
    parser.add_argument("--output", type=Path, default=Path("ear_landmarks.jpg"))
    parser.add_argument("--json", type=Path, default=Path("ear_landmarks.json"))
    parser.add_argument("--input-size", type=int, default=368, help="Square inference size for dynamic-input models")
    parser.add_argument(
        "--crop", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
        help="Ear ROI in original-image pixels; predictions are mapped back to the original image",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Minimum heatmap peak value"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Input image not found: {args.image}")
    if not args.model.is_file():
        raise SystemExit(
            f"Model weights not found: {args.model}\n"
            "The upstream repository does not include pretrained weights. "
            "Train train_face.py followed by train_face2ear.py, or obtain "
            "saved_model_openpose_face2ear_v1.h5 from the author."
        )

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"OpenCV could not decode: {args.image}")

    model = tf.keras.models.load_model(args.model, compile=False)
    if args.weights:
        model.load_weights(args.weights)
    # Combined face-to-ear models name the ear head ``s6_e``; standalone ear
    # models produced by train_ear.py name it ``s6``.
    for layer_name in ("s6_e", "s6"):
        try:
            ear_output = model.get_layer(layer_name).output
            break
        except ValueError:
            continue
    else:
        raise SystemExit("The model has neither an 's6_e' nor 's6' ear-landmark output layer.")
    predictor = tf.keras.Model(model.input, ear_output)

    input_shape = model.input_shape
    if input_shape[1] is None and input_shape[2] is None:
        size = args.input_size
    elif input_shape[1] == input_shape[2]:
        size = int(input_shape[1])
    else:
        raise SystemExit(f"Expected a fixed square or fully dynamic model input, got {input_shape}")
    offset_x = offset_y = 0
    inference_bgr = bgr
    if args.crop:
        x1, y1, x2, y2 = args.crop
        if not (0 <= x1 < x2 <= bgr.shape[1] and 0 <= y1 < y2 <= bgr.shape[0]):
            raise SystemExit(f"Invalid --crop {args.crop} for image size {bgr.shape[1]}x{bgr.shape[0]}")
        inference_bgr = bgr[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    rgb = cv2.cvtColor(inference_bgr, cv2.COLOR_BGR2RGB)
    # Match train_ear.py, which resizes crops directly rather than letterboxing.
    roi_height, roi_width = rgb.shape[:2]
    network_image = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    heatmaps = predictor(network_image[None].astype(np.float32) / 255.0, training=False)[0].numpy()

    map_height, map_width, landmark_count = heatmaps.shape
    if landmark_count != 55:
        raise SystemExit(f"Expected 55 ear heatmaps, got {landmark_count}")

    points: list[dict[str, float | int | bool]] = []
    valid_xy: list[tuple[int, int]] = []
    for index in range(landmark_count):
        smoothed = cv2.GaussianBlur(heatmaps[:, :, index], (5, 5), 2.5)
        flat_index = int(np.argmax(smoothed))
        map_y, map_x = divmod(flat_index, map_width)
        confidence = float(heatmaps[map_y, map_x, index])
        x = int(round((map_x + 0.5) * roi_width / map_width)) + offset_x
        y = int(round((map_y + 0.5) * roi_height / map_height)) + offset_y
        present = (
            confidence >= args.threshold
            and 0 <= x < bgr.shape[1]
            and 0 <= y < bgr.shape[0]
        )
        points.append(
            {"index": index, "x": x, "y": y, "confidence": confidence, "present": present}
        )
        if present:
            valid_xy.append((x, y))

    annotated = bgr.copy()
    offset = 0
    for part_size, color in zip(EAR_PART_SIZES, EAR_PART_COLORS):
        part = points[offset : offset + part_size]
        part_xy = np.asarray(
            [(p["x"], p["y"]) for p in part if p["present"]], dtype=np.int32
        )
        if len(part_xy) == part_size:
            cv2.polylines(annotated, [part_xy], False, color, 2)
        offset += part_size

    ear = None
    if valid_xy:
        coordinates = np.asarray(valid_xy)
        x1, y1 = coordinates.min(axis=0).tolist()
        x2, y2 = coordinates.max(axis=0).tolist()
        ear = {
            "center": {"x": int(round((x1 + x2) / 2)), "y": int(round((y1 + y2) / 2))},
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise SystemExit(f"Could not write output image: {args.output}")
    payload = {"image": str(args.image), "crop": args.crop, "ear": ear, "landmarks": points}
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"ear": ear, "annotated_image": str(args.output), "json": str(args.json)}))


if __name__ == "__main__":
    main()
