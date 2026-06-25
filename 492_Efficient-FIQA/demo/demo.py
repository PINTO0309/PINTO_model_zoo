import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def resize_short_edge(image, size):
    width, height = image.size
    if width <= height:
        new_width = size
        new_height = int(size * height / width)
    else:
        new_height = size
        new_width = int(size * width / height)

    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def resize_to_cover(image, height, width):
    original_width, original_height = image.size
    scale = max(width / original_width, height / original_height)
    new_width = max(width, int(original_width * scale))
    new_height = max(height, int(original_height * scale))

    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def center_crop(image, height, width):
    image_width, image_height = image.size
    left = int(round((image_width - width) / 2.0))
    top = int(round((image_height - height) / 2.0))
    return image.crop((left, top, left + width, top + height))


def preprocess_image(image_path, height, width, use_short_edge_resize):
    image = Image.open(image_path).convert("RGB")

    if use_short_edge_resize and height == width:
        image = resize_short_edge(image, height)
    else:
        image = resize_to_cover(image, height, width)

    image = center_crop(image, height, width)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = (image_array - MEAN) / STD
    image_array = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(image_array, axis=0).astype(np.float32)


def get_providers(backend, gpu_id):
    available_providers = ort.get_available_providers()
    provider_options = {"device_id": str(gpu_id)}

    if backend == "cpu":
        primary_provider = "CPUExecutionProvider"
        providers = [primary_provider]
    elif backend == "cuda":
        primary_provider = "CUDAExecutionProvider"
        providers = [(primary_provider, provider_options)]
    elif backend == "tensorrt":
        primary_provider = "TensorrtExecutionProvider"
        providers = [
            (
                primary_provider,
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": ".",
                    "trt_fp16_enable": True,
                    "trt_op_types_to_exclude": "NonMaxSuppression,NonZero,RoiAlign",
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if primary_provider not in available_providers:
        available = ", ".join(available_providers)
        raise RuntimeError(
            f"{primary_provider} is not available. "
            f"Available providers: {available}. "
            "Install an ONNX Runtime package that supports the requested backend."
        )

    return providers


def get_model_shape(session):
    model_input = session.get_inputs()[0]
    shape = model_input.shape
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D input, but got shape: {shape}")
    return shape


def is_dynamic_dim(dim):
    return isinstance(dim, str) or dim is None


def validate_input_shape(model_shape, input_array):
    batch, channels, height, width = input_array.shape
    expected_batch, expected_channels, expected_height, expected_width = model_shape

    if expected_batch != 1 or batch != 1:
        raise ValueError(f"Only batch size 1 is supported. Model={model_shape}, input={input_array.shape}")
    if expected_channels != 3 or channels != 3:
        raise ValueError(f"Only 3-channel RGB input is supported. Model={model_shape}, input={input_array.shape}")

    if not is_dynamic_dim(expected_height) and height != expected_height:
        raise ValueError(f"Input height must be {expected_height}, but got {height}")
    if not is_dynamic_dim(expected_width) and width != expected_width:
        raise ValueError(f"Input width must be {expected_width}, but got {width}")


def predict_quality(onnx_file, image_file, backend, height, width, gpu_id, use_short_edge_resize):
    providers = get_providers(backend, gpu_id)
    session = ort.InferenceSession(str(onnx_file), providers=providers)
    model_shape = get_model_shape(session)

    input_array = preprocess_image(
        image_path=image_file,
        height=height,
        width=width,
        use_short_edge_resize=use_short_edge_resize,
    )
    validate_input_shape(model_shape, input_array)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_array})[0]
    return float(np.asarray(output).reshape(-1)[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Face Image Quality Assessment ONNX demo")
    parser.add_argument(
        "--onnx_file",
        type=str,
        default="onnx/FIQA_EdgeNeXt_XXS_1x3x352x352.onnx",
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "--image_file",
        type=str,
        default="demo_images/z06399.png",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "tensorrt"],
        help="Inference backend.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=352,
        help="Default square input size.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Input crop height. Defaults to image_size.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Input crop width. Defaults to image_size.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU device ID for cuda or tensorrt backend.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    height = args.height if args.height is not None else args.image_size
    width = args.width if args.width is not None else args.image_size
    use_short_edge_resize = args.height is None and args.width is None

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    try:
        score = predict_quality(
            onnx_file=Path(args.onnx_file),
            image_file=Path(args.image_file),
            backend=args.backend,
            height=height,
            width=width,
            gpu_id=args.gpu_id,
            use_short_edge_resize=use_short_edge_resize,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"The quality score of the image {args.image_file} is {score:.4f}")


if __name__ == "__main__":
    main()
