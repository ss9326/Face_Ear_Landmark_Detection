"""Train the standalone 55-point iBUG ear-landmark model.

Examples:
  python train_ear.py --check-data
  python train_ear.py --smoke-test --batch-size 1
  python train_ear.py --batch-size 4 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from model_config.model import model_openpose_a2a_v2


LANDMARK_SIZE = 55
STAGES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/CollectionA"))
    parser.add_argument("--output", type=Path, default=Path("saved_model/saved_model_openpose_ear_v1.h5"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("saved_model/ear_checkpoints"))
    parser.add_argument("--resume", type=Path, help="Optional .weights.h5 checkpoint to load")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=368)
    parser.add_argument("--map-size", type=int, default=100)
    parser.add_argument("--map-sigma", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--validation-steps", type=int)
    parser.add_argument("--smoke-test", action="store_true", help="Train and validate one batch, then save a test model")
    parser.add_argument("--check-data", action="store_true", help="Validate files and one decoded sample without training")
    return parser.parse_args()


def paired_stems(directory: Path) -> list[str]:
    images = {path.stem for path in directory.glob("*.png")}
    annotations = {path.stem for path in directory.glob("*.pts")}
    missing_annotations = sorted(images - annotations)
    missing_images = sorted(annotations - images)
    if missing_annotations or missing_images:
        raise ValueError(
            f"Unpaired files in {directory}: {len(missing_annotations)} image(s) "
            f"lack .pts and {len(missing_images)} annotation(s) lack .png"
        )
    return [str(directory / stem) for stem in sorted(images)]


def read_pts(file_path: bytes, input_shape: np.ndarray) -> np.ndarray:
    points = np.loadtxt(
        file_path.decode("utf-8"), comments=("version:", "n_points:", "{", "}")
    )
    if points.shape != (LANDMARK_SIZE, 2):
        raise ValueError(f"Expected 55x2 points, got {points.shape}")
    norm = np.asarray([input_shape[1] / 2, input_shape[0] / 2])
    return ((points - norm) / norm).astype(np.float32)


def make_confidence_map(label: np.ndarray, map_size: int, sigma: float) -> np.ndarray:
    norm = np.asarray([map_size / 2, map_size / 2])
    points = label * norm + norm
    grid_y, grid_x = np.mgrid[:map_size, :map_size]
    distance = (
        (grid_x[..., None] - points[:, 0]) ** 2
        + (grid_y[..., None] - points[:, 1]) ** 2
    )
    return np.exp(-distance / sigma**2).astype(np.float32)


def crop_around_ear(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = np.asarray([width / 2, height / 2])
    points = label * center + center
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    ear_width, ear_height = maximum - minimum
    x1 = max(int(minimum[0] - ear_width * 10), 0)
    x2 = min(int(minimum[0] + ear_width * 10), width)
    y1 = max(int(minimum[1] - ear_height * 10), 0)
    y2 = min(int(minimum[1] + ear_height * 10), height)
    cropped = image[y1:y2, x1:x2]
    shifted = points - np.asarray([x1, y1])
    new_height, new_width = cropped.shape[:2]
    new_center = np.asarray([new_width / 2, new_height / 2])
    return cropped, ((shifted - new_center) / new_center).astype(np.float32)


def shrink_ear_width(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = np.asarray([width / 2, height / 2])
    points = label * center + center
    ratio = np.random.randint(1, 4)
    min_x, max_x = int(points[:, 0].min()), int(points[:, 0].max())
    shrunk = np.concatenate(
        (image[:, :min_x], image[:, min_x:max_x:ratio], image[:, max_x:]), axis=1
    )
    points = (points - np.asarray([min_x, 0])) / np.asarray([ratio, 1]) + np.asarray([min_x, 0])
    new_height, new_width = shrunk.shape[:2]
    new_center = np.asarray([new_width / 2, new_height / 2])
    return shrunk, ((points - new_center) / new_center).astype(np.float32)


def make_process_fn(input_size: int, map_size: int, sigma: float, training: bool):
    def process(stem: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.decode_png(tf.io.read_file(tf.strings.join([stem, ".png"])), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        label = tf.numpy_function(read_pts, [tf.strings.join([stem, ".pts"]), tf.shape(image)], tf.float32)
        label.set_shape([LANDMARK_SIZE, 2])
        if training:
            image, label = tf.numpy_function(shrink_ear_width, [image, label], [tf.float32, tf.float32])
            image.set_shape([None, None, 3])
            label.set_shape([LANDMARK_SIZE, 2])
        image, label = tf.numpy_function(crop_around_ear, [image, label], [tf.float32, tf.float32])
        image.set_shape([None, None, 3])
        label.set_shape([LANDMARK_SIZE, 2])
        if training:
            image, label = tf.cond(
                tf.random.uniform(()) < 0.5,
                lambda: (tf.image.flip_left_right(image), label * tf.constant([-1.0, 1.0])),
                lambda: (image, label),
            )
        heatmaps = tf.numpy_function(make_confidence_map, [label, map_size, sigma], tf.float32)
        heatmaps.set_shape([map_size, map_size, LANDMARK_SIZE])
        image = tf.image.resize(image, [input_size, input_size])
        image.set_shape([input_size, input_size, 3])
        return image, heatmaps

    return process


def make_dataset(stems: list[str], batch_size: int, process_fn, training: bool, seed: int) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(stems)
    if training:
        dataset = dataset.shuffle(len(stems), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(process_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def stage_loss(map_size: int):
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.tile(tf.not_equal(y_true, 0), [1, 1, 1, STAGES])
        target = tf.tile(y_true, [1, 1, 1, STAGES])
        prediction = tf.image.resize(y_pred, [map_size, map_size])
        return tf.reduce_mean(
            tf.square(target - prediction) * tf.cast(mask, tf.float32), axis=-1
        )

    return loss


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.epochs < 1:
        raise SystemExit("--batch-size and --epochs must be positive")
    tf.keras.utils.set_random_seed(args.seed)

    train_stems = paired_stems(args.data_dir / "train")
    test_stems = paired_stems(args.data_dir / "test")
    if not train_stems or not test_stems:
        raise SystemExit(f"No paired PNG/PTS data found below {args.data_dir}")
    print(json.dumps({"train_samples": len(train_stems), "test_samples": len(test_stems)}))

    train_process = make_process_fn(args.input_size, args.map_size, args.map_sigma, True)
    test_process = make_process_fn(args.input_size, args.map_size, args.map_sigma, False)
    train_dataset = make_dataset(train_stems, args.batch_size, train_process, True, args.seed)
    test_dataset = make_dataset(test_stems, args.batch_size, test_process, False, args.seed)
    sample_image, sample_heatmaps = next(iter(test_dataset.unbatch().take(1)))
    print(f"sample image={sample_image.shape}, heatmaps={sample_heatmaps.shape}")
    if args.check_data:
        return

    model = model_openpose_a2a_v2(INPUT_SIZE=args.input_size, LANDMARK_SIZE=LANDMARK_SIZE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=stage_loss(args.map_size),
    )
    if args.resume:
        model.load_weights(args.resume)
        print(f"Loaded weights from {args.resume}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    steps = args.steps_per_epoch or math.ceil(len(train_stems) / args.batch_size)
    validation_steps = args.validation_steps or math.ceil(len(test_stems) / args.batch_size)
    epochs = args.epochs
    if args.smoke_test:
        steps = validation_steps = epochs = 1

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(args.checkpoint_dir / "training.csv", append=True),
        tf.keras.callbacks.BackupAndRestore(args.checkpoint_dir / "backup"),
        tf.keras.callbacks.ModelCheckpoint(
            args.checkpoint_dir / "best.weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]
    model.fit(
        train_dataset.repeat(),
        validation_data=test_dataset.repeat(),
        epochs=epochs,
        steps_per_epoch=steps,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
