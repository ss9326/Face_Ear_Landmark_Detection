"""Run ear-landmark TFLite inference on a yaw-aware MediaPipe face ROI."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
NOSE_TIP = 1
EAR_PART_SIZES = (20, 15, 15, 5)
EAR_PART_COLORS = ((0, 255, 0), (255, 51, 0), (255, 204, 0), (255, 204, 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--tflite", type=Path, default=Path("saved_model/ear_landmarks_float16.tflite"))
    parser.add_argument("--face-model", type=Path, default=Path("saved_model/face_landmarker.task"))
    parser.add_argument("--output", type=Path, default=Path("results/mediapipe_ear.jpg"))
    parser.add_argument("--roi-output", type=Path, default=Path("results/mediapipe_roi.jpg"))
    parser.add_argument("--json", type=Path, default=Path("results/mediapipe_ear.json"))
    parser.add_argument(
        "--horizontal-padding", type=float, default=0.10,
        help="Base padding added to both sides, as a fraction of face width",
    )
    parser.add_argument(
        "--yaw-padding", type=float, default=0.75,
        help="Extra padding on the visible-ear side: fraction of face width times absolute yaw",
    )
    parser.add_argument(
        "--top-padding", type=float, default=0.45,
        help="Padding above the mesh, as a fraction of face height",
    )
    parser.add_argument(
        "--bottom-padding", type=float, default=0.22,
        help="Padding below the mesh, as a fraction of face height",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--face-confidence", type=float, default=0.5)
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args()


def ensure_face_model(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe Face Landmarker to {path} ...")
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def detect_largest_face(rgb: np.ndarray, model_path: Path, confidence: float) -> np.ndarray:
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=5,
        min_face_detection_confidence=confidence,
        min_face_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)
    if not result.face_landmarks:
        raise SystemExit("MediaPipe did not detect a face")
    height, width = rgb.shape[:2]
    faces = [
        np.asarray([(point.x * width, point.y * height) for point in landmarks], dtype=np.float32)
        for landmarks in result.face_landmarks
    ]
    return max(faces, key=lambda points: float(np.prod(points.max(axis=0) - points.min(axis=0))))


def expanded_roi(
    points: np.ndarray,
    image_width: int,
    image_height: int,
    horizontal_padding: float,
    yaw_padding: float,
    top_padding: float,
    bottom_padding: float,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], float, str]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    face_width, face_height = maximum - minimum
    face_center_x = (minimum[0] + maximum[0]) / 2
    # Nose left of the cheek midpoint means the head points toward image-left;
    # the more exposed ear is therefore on image-right (and vice versa).
    yaw = float(np.clip((points[NOSE_TIP, 0] - face_center_x) / max(face_width / 2, 1.0), -1.0, 1.0))
    visible_side = "right" if yaw < 0 else "left" if yaw > 0 else "frontal"

    left_pad = right_pad = horizontal_padding * face_width
    yaw_extra = yaw_padding * face_width * abs(yaw)
    if yaw < 0:
        right_pad += yaw_extra
    elif yaw > 0:
        left_pad += yaw_extra
    x1 = max(0, int(np.floor(minimum[0] - left_pad)))
    y1 = max(0, int(np.floor(minimum[1] - top_padding * face_height)))
    x2 = min(image_width, int(np.ceil(maximum[0] + right_pad)))
    y2 = min(image_height, int(np.ceil(maximum[1] + bottom_padding * face_height)))
    face_box = tuple(int(round(value)) for value in (*minimum, *maximum))
    # Reorder min-x,min-y,max-x,max-y after flattening the two xy vectors.
    face_box = (face_box[0], face_box[1], face_box[2], face_box[3])
    if x2 <= x1 or y2 <= y1:
        raise SystemExit("Computed MediaPipe ROI is empty")
    return (x1, y1, x2, y2), face_box, yaw, visible_side


def run_tflite(model_path: Path, rgb_roi: np.ndarray, threads: int) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    shape = input_detail["shape"].tolist()
    if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
        raise SystemExit(f"Unsupported TFLite input shape: {shape}")
    resized = cv2.resize(rgb_roi, (shape[2], shape[1]), interpolation=cv2.INTER_AREA)
    value = resized[None].astype(np.float32) / 255.0
    if input_detail["dtype"] != np.float32:
        scale, zero_point = input_detail["quantization"]
        value = np.clip(np.round(value / scale + zero_point), -128, 127).astype(input_detail["dtype"])
    interpreter.set_tensor(input_detail["index"], value)
    interpreter.invoke()
    heatmaps = interpreter.get_tensor(output_detail["index"])[0]
    if np.issubdtype(heatmaps.dtype, np.integer):
        scale, zero_point = output_detail["quantization"]
        heatmaps = (heatmaps.astype(np.float32) - zero_point) * scale
    if not np.isfinite(heatmaps).all():
        raise SystemExit("TFLite produced NaN/Inf heatmap values")
    return heatmaps.astype(np.float32)


def decode_heatmaps(
    heatmaps: np.ndarray,
    roi: tuple[int, int, int, int],
    threshold: float,
) -> list[dict[str, float | int | bool]]:
    x1, y1, x2, y2 = roi
    roi_width, roi_height = x2 - x1, y2 - y1
    map_height, map_width, channels = heatmaps.shape
    if channels != 55:
        raise SystemExit(f"Expected 55 heatmaps, got {channels}")
    decoded = []
    for index in range(channels):
        smoothed = cv2.GaussianBlur(heatmaps[:, :, index], (5, 5), 2.5)
        map_y, map_x = divmod(int(np.argmax(smoothed)), map_width)
        confidence = float(heatmaps[map_y, map_x, index])
        decoded.append(
            {
                "index": index,
                "x": int(round(x1 + (map_x + 0.5) * roi_width / map_width)),
                "y": int(round(y1 + (map_y + 0.5) * roi_height / map_height)),
                "confidence": confidence,
                "present": confidence >= threshold,
            }
        )
    return decoded


def draw_result(
    bgr: np.ndarray,
    points: list[dict[str, float | int | bool]],
    roi: tuple[int, int, int, int],
    face_box: tuple[int, int, int, int],
) -> tuple[np.ndarray, dict[str, object] | None]:
    annotated = bgr.copy()
    cv2.rectangle(annotated, roi[:2], roi[2:], (0, 165, 255), 2)
    cv2.rectangle(annotated, face_box[:2], face_box[2:], (255, 255, 0), 2)
    offset = 0
    valid = []
    for part_size, color in zip(EAR_PART_SIZES, EAR_PART_COLORS):
        part = points[offset : offset + part_size]
        for point in part:
            if point["present"]:
                xy = (int(point["x"]), int(point["y"]))
                valid.append(xy)
                cv2.circle(annotated, xy, 2, color, -1)
        for first, second in zip(part, part[1:]):
            if first["present"] and second["present"]:
                cv2.line(
                    annotated,
                    (int(first["x"]), int(first["y"])),
                    (int(second["x"]), int(second["y"])),
                    color,
                    2,
                )
        offset += part_size
    ear = None
    if valid:
        coordinates = np.asarray(valid)
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        cv2.rectangle(annotated, tuple(minimum), tuple(maximum), (0, 255, 255), 2)
        ear = {
            "center": {"x": int(round((minimum[0] + maximum[0]) / 2)), "y": int(round((minimum[1] + maximum[1]) / 2))},
            "bbox": {"x1": int(minimum[0]), "y1": int(minimum[1]), "x2": int(maximum[0]), "y2": int(maximum[1])},
        }
    return annotated, ear


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")
    if not args.tflite.is_file():
        raise SystemExit(f"TFLite model not found: {args.tflite}")
    if min(args.horizontal_padding, args.yaw_padding, args.top_padding, args.bottom_padding) < 0:
        raise SystemExit("ROI padding factors cannot be negative")
    if not 0 <= args.confidence_threshold <= 1:
        raise SystemExit("--confidence-threshold must be between 0 and 1")
    ensure_face_model(args.face_model)
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"OpenCV could not decode: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    face_points = detect_largest_face(rgb, args.face_model, args.face_confidence)
    roi, face_box, yaw, visible_side = expanded_roi(
        face_points,
        bgr.shape[1],
        bgr.shape[0],
        args.horizontal_padding,
        args.yaw_padding,
        args.top_padding,
        args.bottom_padding,
    )
    x1, y1, x2, y2 = roi
    heatmaps = run_tflite(args.tflite, rgb[y1:y2, x1:x2], args.num_threads)
    points = decode_heatmaps(heatmaps, roi, args.confidence_threshold)
    annotated, ear = draw_result(bgr, points, roi, face_box)

    roi_debug = bgr.copy()
    cv2.rectangle(roi_debug, face_box[:2], face_box[2:], (255, 255, 0), 2)
    cv2.rectangle(roi_debug, roi[:2], roi[2:], (0, 165, 255), 3)
    cv2.putText(roi_debug, f"yaw={yaw:+.3f} visible={visible_side}", (roi[0] + 8, max(25, roi[1] + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    for path in (args.output, args.roi_output, args.json):
        path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated) or not cv2.imwrite(str(args.roi_output), roi_debug):
        raise SystemExit("Could not write output images")
    payload = {
        "image": str(args.image),
        "tflite": str(args.tflite),
        "face_bbox": dict(zip(("x1", "y1", "x2", "y2"), face_box)),
        "roi": dict(zip(("x1", "y1", "x2", "y2"), roi)),
        "yaw_score": yaw,
        "visible_ear_side": visible_side,
        "padding": {
            "horizontal": args.horizontal_padding,
            "yaw": args.yaw_padding,
            "top": args.top_padding,
            "bottom": args.bottom_padding,
        },
        "confidence_threshold": args.confidence_threshold,
        "accepted_landmarks": sum(bool(point["present"]) for point in points),
        "ear": ear,
        "landmarks": points,
    }
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("face_bbox", "roi", "yaw_score", "visible_ear_side", "accepted_landmarks", "ear")}, indent=2))
    print(f"Annotated: {args.output}\nROI diagnostic: {args.roi_output}\nJSON: {args.json}")


if __name__ == "__main__":
    main()
