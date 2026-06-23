import argparse
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_IMAGE = "data/test.png"
DEFAULT_DETECTOR = "resources/models/yolomit_t_wholebody28_1x3x480x640.onnx"
DEFAULT_ICAONET_ONNX = "resources/models/icaonet_1x3x160x160.onnx"
DEFAULT_ICAONET_TFLITE = (
    "resources/models/saved_model_icaonet_1x3x160x160/"
    "icaonet_1x3x160x160_float32.tflite"
)
DEFAULT_CROP_OUTPUT = "data/test_160x160.png"
HEAD_CLASS_ID = 7
ICAONET_SIZE = (160, 160)
REQUIREMENTS = (
    (2, "Blurred"),
    (3, "Looking away"),
    (4, "Ink marked/creased"),
    (5, "Unnatural skin tone"),
    (6, "Too dark/light"),
    (7, "Washed out"),
    (8, "Pixelation"),
    (9, "Hair across eyes"),
    (10, "Eyes closed"),
    (11, "Varied background"),
    (12, "Roll/pitch/yaw rotations greater than a predefined thresholds"),
    (13, "Flash reflection on skin"),
    (14, "Red eyes"),
    (15, "Shadows behind head"),
    (16, "Shadows across face"),
    (17, "Dark tinted lenses"),
    (18, "Flash reflection on lenses"),
    (19, "Frames too heavy"),
    (20, "Frame covering eyes"),
    (21, "Hat/cap"),
    (22, "Veil over face"),
    (23, "Mouth open"),
    (24, "Presence of other faces or toys too close to face"),
)


@dataclass
class Detection:
    score: float
    xyxy: np.ndarray


@dataclass
class CropResult:
    image_rgb: np.ndarray
    square_xyxy: np.ndarray


def load_rgb_image(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def get_nchw_input_size(session: ort.InferenceSession) -> Tuple[int, int]:
    shape = session.get_inputs()[0].shape
    if len(shape) != 4:
        raise ValueError(f"Expected NCHW input shape, got: {shape}")
    height, width = shape[2], shape[3]
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError(f"Detector input shape must be static NCHW, got: {shape}")
    return height, width


def prepare_yolo_input(image_rgb: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    return np.transpose(tensor, (2, 0, 1))[np.newaxis]


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    boxes_xyxy = np.empty_like(boxes_xywh, dtype=np.float32)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
    return boxes_xyxy


def clip_boxes(boxes_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = boxes_xyxy.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
    return clipped


def nms_xyxy(
    boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> List[int]:
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        current = order[0]
        keep.append(int(current))
        if order.size == 1:
            break

        others = order[1:]
        xx1 = np.maximum(boxes_xyxy[current, 0], boxes_xyxy[others, 0])
        yy1 = np.maximum(boxes_xyxy[current, 1], boxes_xyxy[others, 1])
        xx2 = np.minimum(boxes_xyxy[current, 2], boxes_xyxy[others, 2])
        yy2 = np.minimum(boxes_xyxy[current, 3], boxes_xyxy[others, 3])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        intersection = inter_w * inter_h

        current_area = (
            (boxes_xyxy[current, 2] - boxes_xyxy[current, 0])
            * (boxes_xyxy[current, 3] - boxes_xyxy[current, 1])
        )
        other_areas = (
            (boxes_xyxy[others, 2] - boxes_xyxy[others, 0])
            * (boxes_xyxy[others, 3] - boxes_xyxy[others, 1])
        )
        union = current_area + other_areas - intersection
        iou = intersection / np.maximum(union, 1e-7)
        order = others[iou <= iou_threshold]

    return keep


def detect_head(
    image_rgb: np.ndarray,
    detector_path: str,
    class_id: int,
    score_threshold: float,
    iou_threshold: float,
) -> Detection:
    session = ort.InferenceSession(detector_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_h, input_w = get_nchw_input_size(session)

    original_h, original_w = image_rgb.shape[:2]
    detector_input = prepare_yolo_input(image_rgb, input_h, input_w)
    output = session.run([output_name], {input_name: detector_input})[0]
    predictions = output[0].T

    if predictions.shape[1] <= 4 + class_id:
        raise ValueError(
            f"class_id={class_id} is out of range for YOLO output shape "
            f"{output.shape}"
        )

    scores = predictions[:, 4 + class_id]
    candidate_mask = scores >= score_threshold
    if not np.any(candidate_mask):
        raise RuntimeError(
            f"No class_id={class_id} detection above score threshold "
            f"{score_threshold:.3f}"
        )

    boxes_xyxy = xywh_to_xyxy(predictions[candidate_mask, :4])
    candidate_scores = scores[candidate_mask].astype(np.float32)

    scale_x = original_w / float(input_w)
    scale_y = original_h / float(input_h)
    boxes_xyxy[:, [0, 2]] *= scale_x
    boxes_xyxy[:, [1, 3]] *= scale_y
    boxes_xyxy = clip_boxes(boxes_xyxy, original_w, original_h)

    keep = nms_xyxy(boxes_xyxy, candidate_scores, iou_threshold)
    if not keep:
        raise RuntimeError("No detection remained after NMS")

    best_idx = keep[0]
    return Detection(
        score=float(candidate_scores[best_idx]),
        xyxy=boxes_xyxy[best_idx].astype(np.float32),
    )


def square_crop_rgb(image_rgb: np.ndarray, xyxy: np.ndarray) -> CropResult:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid crop box: {xyxy.tolist()}")

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(width, height)
    square = np.array(
        [
            center_x - side / 2.0,
            center_y - side / 2.0,
            center_x + side / 2.0,
            center_y + side / 2.0,
        ],
        dtype=np.float32,
    )

    sx1 = int(np.floor(square[0]))
    sy1 = int(np.floor(square[1]))
    sx2 = int(np.ceil(square[2]))
    sy2 = int(np.ceil(square[3]))

    image_h, image_w = image_rgb.shape[:2]
    pad_left = max(0, -sx1)
    pad_top = max(0, -sy1)
    pad_right = max(0, sx2 - image_w)
    pad_bottom = max(0, sy2 - image_h)

    if any((pad_left, pad_top, pad_right, pad_bottom)):
        padded = cv2.copyMakeBorder(
            image_rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )
    else:
        padded = image_rgb

    crop_x1 = sx1 + pad_left
    crop_y1 = sy1 + pad_top
    crop_x2 = sx2 + pad_left
    crop_y2 = sy2 + pad_top
    crop = padded[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        raise RuntimeError(f"Square crop is empty: {square.tolist()}")

    crop = cv2.resize(crop, ICAONET_SIZE, interpolation=cv2.INTER_AREA)
    return CropResult(image_rgb=crop, square_xyxy=square)


def prepare_icaonet_onnx_input(crop_rgb: np.ndarray) -> np.ndarray:
    tensor = crop_rgb.astype(np.float32) / 255.0
    return np.transpose(tensor, (2, 0, 1))[np.newaxis]


def prepare_icaonet_tflite_input(crop_rgb: np.ndarray) -> np.ndarray:
    return (crop_rgb.astype(np.float32) / 255.0)[np.newaxis]


def run_icaonet_onnx(crop_rgb: np.ndarray, model_path: str) -> np.ndarray:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run(
        [output_name], {input_name: prepare_icaonet_onnx_input(crop_rgb)}
    )[0]
    return output.astype(np.float32)


def get_tflite_interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter, "ai_edge_litert"
    except ImportError:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter, "tflite_runtime"


def run_icaonet_tflite(crop_rgb: np.ndarray, model_path: str) -> Tuple[np.ndarray, str]:
    interpreter_class, runtime_name = get_tflite_interpreter_class()
    interpreter = interpreter_class(model_path=model_path)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(
        input_detail["index"], prepare_icaonet_tflite_input(crop_rgb)
    )
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])
    return output.astype(np.float32), runtime_name


def print_requirements(output_reqs: np.ndarray) -> None:
    values = output_reqs.squeeze()
    if values.shape != (len(REQUIREMENTS),):
        raise ValueError(
            f"Expected output_reqs shape compatible with ({len(REQUIREMENTS)},), "
            f"got {output_reqs.shape}"
        )
    for (requirement_id, description), value in zip(REQUIREMENTS, values):
        print(f"[{requirement_id:02}] {description}: {float(value)}")


def print_detection(detection: Detection, crop_result: CropResult) -> None:
    print("Selected Head detection")
    print(f"score: {detection.score:.6f}")
    print(f"xyxy: {detection.xyxy.tolist()}")
    print(f"square_xyxy: {crop_result.square_xyxy.tolist()}")
    print()


def save_crop_rgb(crop_rgb: np.ndarray, output_path: str) -> None:
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(output_path, crop_bgr):
        raise RuntimeError(f"Could not write crop image: {output_path}")


def run(args: argparse.Namespace) -> None:
    image_rgb = load_rgb_image(args.image)
    detection = detect_head(
        image_rgb=image_rgb,
        detector_path=args.detector,
        class_id=args.class_id,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
    )
    crop_result = square_crop_rgb(image_rgb, detection.xyxy)
    if args.crop_output:
        save_crop_rgb(crop_result.image_rgb, args.crop_output)
    print_detection(detection, crop_result)
    if args.crop_output:
        print(f"crop_output: {args.crop_output}")
        print()

    onnx_output = None
    tflite_output = None

    if args.backend in ("onnx", "both"):
        onnx_output = run_icaonet_onnx(crop_result.image_rgb, args.icaonet_onnx)
        print("Backend: onnx")
        print_requirements(onnx_output)
        print()

    if args.backend in ("tflite", "both"):
        tflite_output, runtime_name = run_icaonet_tflite(
            crop_result.image_rgb, args.icaonet_tflite
        )
        print(f"Backend: tflite ({runtime_name})")
        print_requirements(tflite_output)
        print()

    if args.backend == "both" and onnx_output is not None and tflite_output is not None:
        max_abs_diff = np.max(np.abs(onnx_output - tflite_output))
        print(f"max_abs_diff: {max_abs_diff}")


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect classid=7 Head with YOLO, square-crop it without 1.5x "
            "expansion, and run ICAONet ONNX/TFLite inference."
        )
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--detector", default=DEFAULT_DETECTOR)
    parser.add_argument("--backend", choices=("onnx", "tflite", "both"), default="onnx")
    parser.add_argument("--icaonet-onnx", default=DEFAULT_ICAONET_ONNX)
    parser.add_argument("--icaonet-tflite", default=DEFAULT_ICAONET_TFLITE)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--class-id", type=int, default=HEAD_CLASS_ID)
    parser.add_argument(
        "--crop-output",
        default=DEFAULT_CROP_OUTPUT,
        help="Path where the cropped and resized 160x160 image will be saved.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
