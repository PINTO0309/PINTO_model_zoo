#!/usr/bin/env python
import copy
import time
import argparse
from typing import Optional, Union

import cv2
import numpy as np
import onnxruntime  # type:ignore


def run_inference(
    onnx_session: onnxruntime.InferenceSession,
    image: np.ndarray,
    score_th: Optional[float] = None,
) -> np.ndarray:
    """
    Performs inference on an image using the provided ONNX model session and generates a mask.

    Args:
        onnx_session (onnxruntime.InferenceSession): The ONNX runtime session.
        image (np.ndarray): The input image.
        score_th (Optional[float], optional): Threshold for score. If provided, the mask will be binarized based on this threshold. Defaults to None.

    Returns:
        np.ndarray: The processed mask image.
    """
    # Get ONNX input size
    input_size = onnx_session.get_inputs()[0].shape
    input_height, input_width = input_size[1], input_size[2]

    # Preprocess: Resize, convert BGR to RGB, and cast to float32
    input_image = cv2.resize(image, dsize=(input_width, input_height))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.astype('float32') / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    # Perform inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: input_image})

    # Postprocess: Squeeze, normalize, optionally apply threshold, and cast to uint8
    mask = np.squeeze(result[0])
    min_val, max_val = np.min(mask), np.max(mask)
    mask = (mask - min_val) / (max_val - min_val)
    if score_th is not None:
        mask = np.where(mask < score_th, 0, 1)
    mask = (mask * 255).astype('uint8')

    return mask


def main() -> None:
    """
    The main function that parses command-line arguments, initializes video capture,
    loads the ONNX model, performs inference on each video frame, and displays the results.
    """
    parser = argparse.ArgumentParser(
        description="Selfie Segmentation using ONNX model.")

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Camera device number. Default is 0.",
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=None,
        help=
        "Path to a video file. If specified, video will be used instead of camera.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_tflite_tfjs_tftrt_onnx_coreml/model_float32.onnx',
        help="Path to the ONNX model to be used.",
    )
    parser.add_argument(
        "--score_th",
        type=float,
        default=None,
        help=
        "Score threshold for mask binarization. If specified, the mask will be binarized based on this threshold.",
    )

    args = parser.parse_args()
    model_path = args.model
    score_th = args.score_th

    # Initialize video capture
    cap_device: Union[int, str] = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap = cv2.VideoCapture(cap_device)

    # Load the ONNX model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

    while True:
        start_time = time.time()

        # Read a frame from the video capture
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Execute inference
        mask = run_inference(
            onnx_session,
            frame,
            score_th,
        )

        elapsed_time = time.time() - start_time

        # Resize the mask to match the original image size using nearest neighbor interpolation
        mask_resized = cv2.resize(
            mask,
            dsize=(debug_image.shape[1], debug_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Optional: Ensure the mask is binary after resizing
        _, mask_binary = cv2.threshold(
            mask_resized,
            127,
            255,
            cv2.THRESH_BINARY,
        )

        # Overlay the mask on the original image using bitwise operations
        mask_inv = cv2.bitwise_not(mask_binary)
        foreground = cv2.bitwise_and(
            debug_image,
            debug_image,
            mask=mask_binary,
        )
        background = cv2.bitwise_and(
            np.full(debug_image.shape, 255, dtype=np.uint8),
            np.full(debug_image.shape, 255, dtype=np.uint8),
            mask=mask_inv,
        )
        mask_image = cv2.add(foreground, background)

        # Display the elapsed inference time
        elapsed_time_text = f"Elapsed time: {round(elapsed_time * 1000, 1)} ms"
        cv2.putText(debug_image, elapsed_time_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Handle key inputs
        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            break

        # Display the images in separate windows
        cv2.imshow('Selfie Segmentation: Input', debug_image)
        cv2.imshow('Selfie Segmentation: Mask Overlay', mask_image)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
