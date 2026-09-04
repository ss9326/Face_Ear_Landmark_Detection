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
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
EAR_PART_SIZES = (20, 15, 15, 5)
EAR_PART_COLORS = ((0, 255, 0), (255, 51, 0), (255, 204, 0), (255, 204, 0))
# Reference dimension the constants below are tuned for (my_photo_1..4, 2736x3648).
REFERENCE_DIMENSION = 3648.0
BOX_THICKNESS = 6
LANDMARK_RADIUS = 5
LANDMARK_LINE_THICKNESS = 2


def drawing_scale(bgr: np.ndarray) -> float:
    return max(bgr.shape[0], bgr.shape[1]) / REFERENCE_DIMENSION


def scaled(value: int, scale: float) -> int:
    return max(1, round(value * scale))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images", type=Path, nargs="+",
        help="Image files and/or directories of images to process",
    )
    parser.add_argument("--tflite", type=Path, default=Path("saved_model/ear_landmarks_float16.tflite"))
    parser.add_argument("--face-model", type=Path, default=Path("saved_model/face_landmarker.task"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Directory to write <stem>_mp_ear.jpg / _mp_roi.jpg / _mp_ear.json into",
    )
    parser.add_argument(
        "--horizontal-padding", type=float, default=0.10,
        help="Base padding added to both sides, as a fraction of face width",
    )
    parser.add_argument(
        "--yaw-padding", type=float, default=0.75,
        help="Extra padding on the visible-ear side: fraction of face width times absolute yaw",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--face-confidence", type=float, default=0.5)
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args()


def resolve_images(paths: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            if not found:
                print(f"[{path}] no images found in directory, skipping")
            resolved.extend(found)
        elif path.is_file():
            resolved.append(path)
        else:
            print(f"[{path}] not found, skipping")
    return resolved


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


def create_face_landmarker(model_path: Path, confidence: float) -> mp.tasks.vision.FaceLandmarker:
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=5,
        min_face_detection_confidence=confidence,
        min_face_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def detect_largest_face(landmarker: mp.tasks.vision.FaceLandmarker, rgb: np.ndarray) -> np.ndarray:
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect(image)
    if not result.face_landmarks:
        raise RuntimeError("MediaPipe did not detect a face")
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
    y1 = max(0, int(np.floor(minimum[1])))
    x2 = min(image_width, int(np.ceil(maximum[0] + right_pad)))
    y2 = min(image_height, int(np.ceil(maximum[1])))
    face_box = tuple(int(round(value)) for value in (*minimum, *maximum))
    # Reorder min-x,min-y,max-x,max-y after flattening the two xy vectors.
    face_box = (face_box[0], face_box[1], face_box[2], face_box[3])
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("Computed MediaPipe ROI is empty")
    return (x1, y1, x2, y2), face_box, yaw, visible_side


def letterbox_resize(
    image: np.ndarray, target_width: int, target_height: int
) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    pad_x, pad_y = (target_width - new_width) // 2, (target_height - new_height) // 2
    padded = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
    padded[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return padded, scale, pad_x, pad_y


def create_tflite_interpreter(
    model_path: Path, threads: int
) -> tuple[tf.lite.Interpreter, dict, dict, int, int]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_shape = input_detail["shape"].tolist()
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[3] != 3:
        raise SystemExit(f"Unsupported TFLite input shape: {input_shape}")
    output_shape = output_detail["shape"].tolist()
    if len(output_shape) != 4 or output_shape[0] != 1 or output_shape[3] != 55:
        raise SystemExit(f"Unsupported TFLite output shape: {output_shape}")
    return interpreter, input_detail, output_detail, input_shape[2], input_shape[1]


def run_tflite(
    interpreter: tf.lite.Interpreter,
    input_detail: dict,
    output_detail: dict,
    rgb_roi: np.ndarray,
    input_width: int,
    input_height: int,
) -> tuple[np.ndarray, float, int, int]:
    # The ROI is deliberately non-square (same height as the face box, only
    # expanded sideways); letterbox rather than plain-resize it so it isn't
    # anisotropically stretched to fit the model's fixed square input.
    letterboxed, letterbox_scale, pad_x, pad_y = letterbox_resize(rgb_roi, input_width, input_height)
    value = letterboxed[None].astype(np.float32) / 255.0
    if input_detail["dtype"] != np.float32:
        quant_scale, zero_point = input_detail["quantization"]
        value = np.clip(np.round(value / quant_scale + zero_point), -128, 127).astype(input_detail["dtype"])
    interpreter.set_tensor(input_detail["index"], value)
    interpreter.invoke()
    heatmaps = interpreter.get_tensor(output_detail["index"])[0]
    if np.issubdtype(heatmaps.dtype, np.integer):
        quant_scale, zero_point = output_detail["quantization"]
        heatmaps = (heatmaps.astype(np.float32) - zero_point) * quant_scale
    if not np.isfinite(heatmaps).all():
        raise RuntimeError("TFLite produced NaN/Inf heatmap values")
    return heatmaps.astype(np.float32), letterbox_scale, pad_x, pad_y


def decode_heatmaps(
    heatmaps: np.ndarray,
    roi: tuple[int, int, int, int],
    threshold: float,
    letterbox_scale: float,
    pad_x: int,
    pad_y: int,
    input_width: int,
    input_height: int,
) -> list[dict[str, float | int | bool]]:
    x1, y1, x2, y2 = roi
    roi_width, roi_height = x2 - x1, y2 - y1
    map_height, map_width, channels = heatmaps.shape
    decoded = []
    for index in range(channels):
        smoothed = cv2.GaussianBlur(heatmaps[:, :, index], (5, 5), 2.5)
        map_y, map_x = divmod(int(np.argmax(smoothed)), map_width)
        confidence = float(heatmaps[map_y, map_x, index])
        # Position within the padded square model input, then undo the
        # letterbox to land back in the (non-square) ROI's own coordinates.
        input_x = (map_x + 0.5) * input_width / map_width
        input_y = (map_y + 0.5) * input_height / map_height
        roi_x = (input_x - pad_x) / letterbox_scale
        roi_y = (input_y - pad_y) / letterbox_scale
        # A peak that decodes into the letterbox padding band isn't a real
        # detection; the model never saw actual ROI content there.
        in_content = 0 <= roi_x <= roi_width and 0 <= roi_y <= roi_height
        decoded.append(
            {
                "index": index,
                "x": int(round(x1 + roi_x)),
                "y": int(round(y1 + roi_y)),
                "confidence": confidence,
                "present": confidence >= threshold and in_content,
            }
        )
    return decoded


def draw_result(
    bgr: np.ndarray,
    points: list[dict[str, float | int | bool]],
    roi: tuple[int, int, int, int],
    face_box: tuple[int, int, int, int],
) -> tuple[np.ndarray, dict[str, object] | None]:
    scale = drawing_scale(bgr)
    box_thickness = scaled(BOX_THICKNESS, scale)
    radius = scaled(LANDMARK_RADIUS, scale)
    line_thickness = scaled(LANDMARK_LINE_THICKNESS, scale)
    annotated = bgr.copy()
    cv2.rectangle(annotated, roi[:2], roi[2:], (0, 165, 255), box_thickness)
    cv2.rectangle(annotated, face_box[:2], face_box[2:], (255, 255, 0), box_thickness)
    offset = 0
    valid = []
    for part_size, color in zip(EAR_PART_SIZES, EAR_PART_COLORS):
        part = points[offset : offset + part_size]
        for first, second in zip(part, part[1:]):
            if first["present"] and second["present"]:
                cv2.line(
                    annotated,
                    (int(first["x"]), int(first["y"])),
                    (int(second["x"]), int(second["y"])),
                    color,
                    line_thickness,
                )
        for point in part:
            if point["present"]:
                xy = (int(point["x"]), int(point["y"]))
                valid.append(xy)
                cv2.circle(annotated, xy, radius, color, -1)
        offset += part_size
    ear = None
    if valid:
        coordinates = np.asarray(valid)
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        cv2.rectangle(annotated, tuple(minimum), tuple(maximum), (0, 255, 255), box_thickness)
        ear = {
            "center": {"x": int(round((minimum[0] + maximum[0]) / 2)), "y": int(round((minimum[1] + maximum[1]) / 2))},
            "bbox": {"x1": int(minimum[0]), "y1": int(minimum[1]), "x2": int(maximum[0]), "y2": int(maximum[1])},
        }
    return annotated, ear


def process_image(
    image_path: Path,
    args: argparse.Namespace,
    landmarker: mp.tasks.vision.FaceLandmarker,
    interpreter: tf.lite.Interpreter,
    input_detail: dict,
    output_detail: dict,
    input_width: int,
    input_height: int,
) -> dict:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"OpenCV could not decode: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    face_points = detect_largest_face(landmarker, rgb)
    roi, face_box, yaw, visible_side = expanded_roi(
        face_points,
        bgr.shape[1],
        bgr.shape[0],
        args.horizontal_padding,
        args.yaw_padding,
    )
    x1, y1, x2, y2 = roi
    heatmaps, letterbox_scale, pad_x, pad_y = run_tflite(
        interpreter, input_detail, output_detail, rgb[y1:y2, x1:x2], input_width, input_height
    )
    points = decode_heatmaps(
        heatmaps, roi, args.confidence_threshold, letterbox_scale, pad_x, pad_y, input_width, input_height
    )
    annotated, ear = draw_result(bgr, points, roi, face_box)

    roi_debug = bgr.copy()
    roi_debug_thickness = scaled(BOX_THICKNESS, drawing_scale(bgr))
    cv2.rectangle(roi_debug, face_box[:2], face_box[2:], (255, 255, 0), roi_debug_thickness)
    cv2.rectangle(roi_debug, roi[:2], roi[2:], (0, 165, 255), roi_debug_thickness)
    cv2.putText(roi_debug, f"yaw={yaw:+.3f} visible={visible_side}", (roi[0] + 8, max(25, roi[1] + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    stem = image_path.stem
    output_path = args.output_dir / f"{stem}_mp_ear.jpg"
    roi_output_path = args.output_dir / f"{stem}_mp_roi.jpg"
    json_path = args.output_dir / f"{stem}_mp_ear.json"
    if not cv2.imwrite(str(output_path), annotated) or not cv2.imwrite(str(roi_output_path), roi_debug):
        raise RuntimeError("Could not write output images")

    accepted_landmarks = sum(bool(point["present"]) for point in points)
    payload = {
        "image": str(image_path),
        "tflite": str(args.tflite),
        "face_bbox": dict(zip(("x1", "y1", "x2", "y2"), face_box)),
        "roi": dict(zip(("x1", "y1", "x2", "y2"), roi)),
        "yaw_score": yaw,
        "visible_ear_side": visible_side,
        "padding": {
            "horizontal": args.horizontal_padding,
            "yaw": args.yaw_padding,
        },
        "confidence_threshold": args.confidence_threshold,
        "accepted_landmarks": accepted_landmarks,
        "ear": ear,
        "landmarks": points,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[{stem}] yaw={yaw:+.3f} visible={visible_side} "
        f"accepted={accepted_landmarks}/{len(points)} -> {output_path}"
    )
    return payload


def main() -> None:
    args = parse_args()
    if not args.tflite.is_file():
        raise SystemExit(f"TFLite model not found: {args.tflite}")
    if min(args.horizontal_padding, args.yaw_padding) < 0:
        raise SystemExit("ROI padding factors cannot be negative")
    if not 0 <= args.confidence_threshold <= 1:
        raise SystemExit("--confidence-threshold must be between 0 and 1")

    images = resolve_images(args.images)
    if not images:
        raise SystemExit("No input images found")

    ensure_face_model(args.face_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    landmarker = create_face_landmarker(args.face_model, args.face_confidence)
    interpreter, input_detail, output_detail, input_width, input_height = create_tflite_interpreter(
        args.tflite, args.num_threads
    )

    succeeded = 0
    try:
        for image_path in images:
            try:
                process_image(
                    image_path, args, landmarker, interpreter, input_detail, output_detail,
                    input_width, input_height,
                )
                succeeded += 1
            except RuntimeError as error:
                print(f"[{image_path.stem}] FAILED: {error}")
    finally:
        landmarker.close()

    print(f"\n{succeeded}/{len(images)} succeeded.")
    if succeeded == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
