import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


YOLO_INPUT_SIZE = (640, 480)
CLASSIFIER_INPUT_SIZE = (256, 256)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Detection:
    score: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Result:
    head_score: float
    sunglasses_logit: float
    sunglasses_score: float
    box: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect Head regions with YOLO and classify each crop with the "
            "sunglasses ONNX classifier."
        )
    )
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="Input image path(s). Omit when using --camera.",
    )
    parser.add_argument(
        "--yolo",
        type=Path,
        default=Path("onnx/yolomit_t_wholebody28_1x3x480x640.onnx"),
        help="YOLO wholebody ONNX model path.",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("onnx/classification_sunglasses_medium.onnx"),
        help="Sunglasses classification ONNX model path.",
    )
    parser.add_argument(
        "--backend",
        choices=["cuda", "cpu", "tensorrt"],
        default="cuda",
        help="ONNX Runtime backend. Defaults to cuda.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=7,
        help="YOLO class id to crop. Defaults to 7 (Head).",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="YOLO confidence threshold. Defaults to 0.25.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="NMS IoU threshold. Defaults to 0.45.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="USB camera device index. Example: --camera 0.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Capture width to request from the camera. Defaults to 640.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Capture height to request from the camera. Defaults to 480.",
    )
    parser.add_argument(
        "--window-name",
        default="Head sunglasses demo",
        help="OpenCV window name for camera visualization.",
    )
    parser.add_argument(
        "--sunglasses-threshold",
        type=float,
        default=0.5,
        help="Score threshold used to color labels in camera mode. Defaults to 0.5.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Horizontally mirror camera frames before inference and visualization.",
    )
    parser.add_argument(
        "--record-output",
        type=Path,
        default=None,
        help=(
            "MP4 output path for camera mode. Defaults to "
            "demo/recordings/camera_<index>_<timestamp>.mp4."
        ),
    )
    return parser.parse_args()


def build_providers(backend: str, model_path: Path) -> list[Any]:
    available = set(ort.get_available_providers())
    requested = backend.lower()

    def warn(message: str) -> None:
        print(f"warning: {message}", file=sys.stderr)

    def cuda_or_cpu(reason: str) -> list[Any]:
        if "CUDAExecutionProvider" in available:
            warn(f"{reason}; falling back to CUDAExecutionProvider")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        warn(f"{reason}; falling back to CPUExecutionProvider")
        return ["CPUExecutionProvider"]

    if requested == "cpu":
        return ["CPUExecutionProvider"]

    if requested == "cuda":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return cuda_or_cpu("CUDAExecutionProvider is not available")

    if requested == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            return cuda_or_cpu("TensorrtExecutionProvider is not available")

        providers: list[Any] = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(model_path.parent),
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(model_path.parent),
                },
            )
        ]
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    raise ValueError(f"Unsupported backend: {backend}")


def session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return options


def create_session(model_path: Path, backend: str) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options(),
        providers=build_providers(backend, model_path),
    )


def provider_backend_name(provider: str) -> str:
    if provider == "TensorrtExecutionProvider":
        return "tensorrt"
    if provider == "CUDAExecutionProvider":
        return "cuda"
    if provider == "CPUExecutionProvider":
        return "cpu"
    return provider


def session_backend_name(session: ort.InferenceSession) -> str:
    providers = session.get_providers()
    if not providers:
        return "unknown"
    return provider_backend_name(providers[0])


def display_backend_name(
    yolo_session: ort.InferenceSession,
    classifier_session: ort.InferenceSession,
) -> str:
    yolo_backend = session_backend_name(yolo_session)
    classifier_backend = session_backend_name(classifier_session)
    if yolo_backend == classifier_backend:
        return yolo_backend
    return f"yolo:{yolo_backend} cls:{classifier_backend}"


def preprocess_yolo(image: Image.Image) -> np.ndarray:
    resized = image.resize(YOLO_INPUT_SIZE)
    tensor = np.asarray(resized, dtype=np.float32) / 255.0
    return tensor.transpose(2, 0, 1)[None]


def preprocess_classifier(crop: Image.Image) -> np.ndarray:
    resized = crop.resize(CLASSIFIER_INPUT_SIZE)
    tensor = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.transpose(2, 0, 1)[None]


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def normalize_yolo_output(output: np.ndarray) -> np.ndarray:
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"Unexpected YOLO output shape: {output.shape}")

    predictions = output[0]
    if predictions.shape[0] == 32:
        return predictions
    if predictions.shape[-1] == 32:
        return predictions.T

    raise ValueError(f"Unexpected YOLO prediction shape: {predictions.shape}")


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes.T
    return np.stack(
        [
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
        ],
        axis=1,
    )


def scale_and_clamp_boxes(boxes: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    image_w, image_h = image_size
    scale = np.array(
        [
            image_w / YOLO_INPUT_SIZE[0],
            image_h / YOLO_INPUT_SIZE[1],
            image_w / YOLO_INPUT_SIZE[0],
            image_h / YOLO_INPUT_SIZE[1],
        ],
        dtype=np.float32,
    )
    boxes = boxes * scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_h)
    return boxes


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        intersection = inter_w * inter_h
        union = areas[i] + areas[rest] - intersection
        ious = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = rest[ious <= iou_threshold]

    return keep


def detect_heads(
    image: Image.Image,
    yolo_session: ort.InferenceSession,
    class_id: int,
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    input_name = yolo_session.get_inputs()[0].name
    output = yolo_session.run(None, {input_name: preprocess_yolo(image)})[0]
    predictions = normalize_yolo_output(output)

    class_index = 4 + class_id
    if class_index >= predictions.shape[0]:
        raise ValueError(
            f"class_id={class_id} is out of range for YOLO output shape {output.shape}"
        )

    scores = predictions[class_index]
    mask = scores >= conf_threshold
    if not np.any(mask):
        return []

    boxes = cxcywh_to_xyxy(predictions[:4, mask].T)
    scores = scores[mask]
    boxes = scale_and_clamp_boxes(boxes, image.size)

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    scores = scores[valid]
    if len(boxes) == 0:
        return []

    keep = nms(boxes, scores, iou_threshold)
    detections: list[Detection] = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        int_box = (
            int(np.floor(x1)),
            int(np.floor(y1)),
            int(np.ceil(x2)),
            int(np.ceil(y2)),
        )
        detections.append(Detection(score=float(scores[i]), box=int_box))

    return detections


def classify_crop(
    crop: Image.Image,
    classifier_session: ort.InferenceSession,
) -> tuple[float, float]:
    input_name = classifier_session.get_inputs()[0].name
    output = classifier_session.run(None, {input_name: preprocess_classifier(crop)})[0]
    logit = float(np.asarray(output).reshape(-1)[0])
    return logit, sigmoid(logit)


def analyze_image(
    image: Image.Image,
    yolo_session: ort.InferenceSession,
    classifier_session: ort.InferenceSession,
    class_id: int,
    conf_threshold: float,
    iou_threshold: float,
) -> list[Result]:
    detections = detect_heads(
        image=image,
        yolo_session=yolo_session,
        class_id=class_id,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    results: list[Result] = []
    for detection in detections:
        crop = image.crop(detection.box)
        logit, score = classify_crop(crop, classifier_session)
        results.append(
            Result(
                head_score=detection.score,
                sunglasses_logit=logit,
                sunglasses_score=score,
                box=detection.box,
            )
        )

    return results


def process_image(
    image_path: Path,
    yolo_session: ort.InferenceSession,
    classifier_session: ort.InferenceSession,
    class_id: int,
    conf_threshold: float,
    iou_threshold: float,
    writer: csv.writer,
) -> None:
    image = Image.open(image_path).convert("RGB")
    results = analyze_image(
        image=image,
        yolo_session=yolo_session,
        classifier_session=classifier_session,
        class_id=class_id,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

    if not results:
        writer.writerow([str(image_path), "no_head"])
        return

    for head_index, result in enumerate(results):
        x1, y1, x2, y2 = result.box
        writer.writerow(
            [
                str(image_path),
                head_index,
                f"{result.head_score:.6f}",
                f"{result.sunglasses_logit:.6f}",
                f"{result.sunglasses_score:.6f}",
                x1,
                y1,
                x2,
                y2,
            ]
        )


def draw_results(
    frame: np.ndarray,
    results: list[Result],
    sunglasses_threshold: float,
    fps: float | None,
    backend_name: str | None,
) -> np.ndarray:
    output = frame.copy()

    def put_outlined_text(
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        font_scale: float,
    ) -> None:
        cv2.putText(
            output,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA,
        )

    for head_index, result in enumerate(results):
        x1, y1, x2, y2 = result.box
        positive = result.sunglasses_score >= sunglasses_threshold
        color = (0, 128, 255) if positive else (0, 200, 0)
        label = (
            f"Head {head_index} "
            f"H:{result.head_score:.2f} "
            f"Sun:{result.sunglasses_score:.2f}"
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label_y = max(20, y1 - 8)
        put_outlined_text(
            label,
            (x1, label_y),
            color,
            0.55,
        )

    status = "Press q or ESC to quit"
    if fps is not None:
        status = f"FPS:{fps:.1f}  {status}"
    put_outlined_text(
        status,
        (10, 24),
        (40, 40, 40),
        0.65,
    )
    next_status_y = 52
    if backend_name is not None:
        put_outlined_text(
            f"Backend: {backend_name}",
            (10, next_status_y),
            (40, 40, 40),
            0.65,
        )
        next_status_y += 28
    if not results:
        put_outlined_text(
            "no_head",
            (10, next_status_y),
            (0, 255, 255),
            0.65,
        )
    return output


def default_record_output(camera_index: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("demo/recordings") / f"camera_{camera_index}_{timestamp}.mp4"


def normalize_record_output(path: Path) -> Path:
    return path if path.suffix.lower() == ".mp4" else path.with_suffix(".mp4")


def create_video_writer(output_path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_h, frame_w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer: {output_path}")
    print(f"recording: {output_path}", file=sys.stderr)
    return writer


def run_camera(
    camera_index: int,
    yolo_session: ort.InferenceSession,
    classifier_session: ort.InferenceSession,
    class_id: int,
    conf_threshold: float,
    iou_threshold: float,
    camera_width: int | None,
    camera_height: int | None,
    window_name: str,
    sunglasses_threshold: float,
    mirror: bool,
    record_output: Path | None,
    backend_name: str,
) -> None:
    capture = cv2.VideoCapture(camera_index)
    if camera_width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    if camera_height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    output_path = normalize_record_output(record_output or default_record_output(camera_index))
    capture_fps = capture.get(cv2.CAP_PROP_FPS)
    record_fps = capture_fps if capture_fps and capture_fps > 0 else 30.0
    video_writer: cv2.VideoWriter | None = None
    last_time = time.perf_counter()
    fps: float | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame from camera index {camera_index}")

            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            results = analyze_image(
                image=image,
                yolo_session=yolo_session,
                classifier_session=classifier_session,
                class_id=class_id,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = current_fps if fps is None else fps * 0.9 + current_fps * 0.1

            display_frame = draw_results(
                frame,
                results,
                sunglasses_threshold,
                fps,
                backend_name,
            )
            if video_writer is None:
                video_writer = create_video_writer(output_path, display_frame, record_fps)
            video_writer.write(display_frame)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        if video_writer is not None:
            video_writer.release()
        capture.release()
        cv2.destroyAllWindows()


def write_header(writer: csv.writer) -> None:
    writer.writerow(
        [
            "image",
            "head_index",
            "head_score",
            "sunglasses_logit",
            "sunglasses_score",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
    )


def main() -> int:
    args = parse_args()
    if args.camera is None and not args.images:
        raise SystemExit("Provide image path(s) or specify --camera <index>.")

    yolo_session = create_session(args.yolo, args.backend)
    classifier_session = create_session(args.classifier, args.backend)

    if args.camera is not None:
        run_camera(
            camera_index=args.camera,
            yolo_session=yolo_session,
            classifier_session=classifier_session,
            class_id=args.class_id,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            window_name=args.window_name,
            sunglasses_threshold=args.sunglasses_threshold,
            mirror=args.mirror,
            record_output=args.record_output,
            backend_name=display_backend_name(yolo_session, classifier_session),
        )
        return 0

    writer = csv.writer(sys.stdout)
    write_header(writer)

    for image_path in args.images:
        process_image(
            image_path=image_path,
            yolo_session=yolo_session,
            classifier_session=classifier_session,
            class_id=args.class_id,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            writer=writer,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
