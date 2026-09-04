"""Export the trained standalone ear heatmap head to LiteRT/TFLite.

The exported model has a fixed NHWC input of ``[1, 368, 368, 3]`` by default.
Pixels must be RGB float32 values in [0, 1].  The output is the 55-channel
``s6`` heatmap tensor; landmark decoding and ROI coordinate mapping stay in the
application.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import train_ear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("saved_model/saved_model_openpose_ear_v1.h5"))
    parser.add_argument("--weights", type=Path, default=Path("saved_model/ear_checkpoints/best.weights.h5"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quantization", choices=("float32", "float16", "int8"), default="float16")
    parser.add_argument("--input-size", type=int, default=368)
    parser.add_argument("--data-dir", type=Path, default=Path("data/CollectionA"))
    parser.add_argument("--representative-samples", type=int, default=100)
    return parser.parse_args()


def load_predictor(model_path: Path, weights_path: Path | None) -> tf.keras.Model:
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    if weights_path:
        if not weights_path.is_file():
            raise SystemExit(f"Weights not found: {weights_path}")
        model.load_weights(weights_path)
    try:
        output = model.get_layer("s6").output
    except ValueError as error:
        raise SystemExit("Expected a standalone ear model with an 's6' layer") from error
    return tf.keras.Model(model.input, output, name="ear_heatmap_predictor")


def representative_dataset(data_dir: Path, input_size: int, count: int):
    stems = train_ear.paired_stems(data_dir / "train")[:count]
    if not stems:
        raise SystemExit(f"No representative training images found in {data_dir / 'train'}")
    process = train_ear.make_process_fn(input_size, 100, 2.5, False)
    dataset = train_ear.make_dataset(stems, 1, process, False, 42)
    for image, _ in dataset:
        yield [image.numpy().astype(np.float32)]


def main() -> None:
    args = parse_args()
    if args.input_size < 1 or args.representative_samples < 1:
        raise SystemExit("--input-size and --representative-samples must be positive")
    output = args.output or Path(f"saved_model/ear_landmarks_{args.quantization}.tflite")
    predictor = load_predictor(args.model, args.weights)

    @tf.function(input_signature=[tf.TensorSpec([1, args.input_size, args.input_size, 3], tf.float32, name="image")])
    def serve(image: tf.Tensor) -> dict[str, tf.Tensor]:
        return {"heatmaps": predictor({"input_layer": image}, training=False)}

    # This legacy HDF5/Keras graph is explicitly frozen because TensorFlow 2.21
    # otherwise converts its weights to uninitialized TFLite resource handles.
    concrete = serve.get_concrete_function()
    frozen = convert_variables_to_constants_v2(concrete)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    if args.quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif args.quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(
            args.data_dir, args.input_size, args.representative_samples
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Float I/O keeps Android preprocessing identical. Quantize/dequantize
        # nodes surround an otherwise integer model.
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    tflite = converter.convert()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(tflite)

    interpreter = tf.lite.Interpreter(model_content=tflite)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    metadata = {
        "model": str(output),
        "quantization": args.quantization,
        "bytes": len(tflite),
        "megabytes": round(len(tflite) / 1024**2, 2),
        "input": {
            "name": input_detail["name"],
            "shape": input_detail["shape"].tolist(),
            "dtype": np.dtype(input_detail["dtype"]).name,
            "color": "RGB",
            "range": [0.0, 1.0],
        },
        "output": {
            "name": output_detail["name"],
            "shape": output_detail["shape"].tolist(),
            "dtype": np.dtype(output_detail["dtype"]).name,
            "channels": 55,
        },
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
