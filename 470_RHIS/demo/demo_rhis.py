#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import argparse

import cv2
import numpy as np
import onnxruntime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="best_model_b1_640x640_80x60_0.8551_dil1.onnx",
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Score threshold for mask visualization.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Transparency for the mask overlay.",
    )
    parser.add_argument(
        "--draw_roi",
        action="store_true",
        help="If specified, draw ROI rectangles on the image.",
    )
    args = parser.parse_args()

    return args


def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def preprocess(image, rois_unnormalized, input_width, input_height):
    # PreProcess:Resize, BGR to RGB, Normalize, To NCHW
    input_image = cv2.resize(image, dsize=(input_width, input_height))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image / 255.0
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0).astype("float32")

    # Normalize each ROI (x1, y1, x2, y2) by dividing by frame size
    rois_normalized = []
    frame_height, frame_width = image.shape[:2]
    for r in rois_unnormalized:
        x1, y1, x2, y2 = r
        nx1 = x1 / frame_width
        ny1 = y1 / frame_height
        nx2 = x2 / frame_width
        ny2 = y2 / frame_height
        rois_normalized.append([0, nx1, ny1, nx2, ny2])
    rois = np.array(rois_normalized, dtype=np.float32)

    return input_image, rois


def postprocess(masks_output, rois_unnormalized, score_threshold):
    person_class_id = 1
    roi_masks = []
    for i in range(masks_output.shape[0]):
        # Get per-ROI logits and convert to per-pixel probabilities
        logits = masks_output[i]  # (C, Hm, Wm)
        probs = softmax(logits, axis=0)  # (C, Hm, Wm)
        pred_class = np.argmax(probs, axis=0)  # (Hm, Wm)
        pred_score = np.max(probs, axis=0)  # (Hm, Wm)

        # Keep "person" pixels above the threshold
        person_mask = (pred_class == person_class_id) & (pred_score > score_threshold)

        # Resize mask to match the ROI box size for direct placement
        x1, y1, x2, y2 = rois_unnormalized[i].astype(int)
        roi_w, roi_h = max(0, x2 - x1), max(0, y2 - y1)
        if roi_w <= 0 or roi_h <= 0:
            # Skip invalid ROI
            continue

        roi_mask = cv2.resize(
            person_mask.astype(np.uint8),
            (roi_w, roi_h),
            interpolation=cv2.INTER_NEAREST,
        )

        roi_masks.append((roi_mask, (x1, y1, x2, y2)))
    return roi_masks


def get_id_color(index):
    temp_index = abs(int(index + 5)) * 3
    color = [(37 * temp_index) % 255, (17 * temp_index) % 255, (29 * temp_index) % 255]
    return color


def visualize(image, roi_masks, alpha, draw_roi):
    result = image.copy()
    image_height, image_width = result.shape[:2]

    for index, (roi_mask, (x1, y1, x2, y2)) in enumerate(roi_masks):
        color = get_id_color(index)

        # Clip ROI to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_width, x2), min(image_height, y2)

        # Skip invalid ROIs
        roi_w, roi_h = x2 - x1, y2 - y1
        if roi_w <= 0 or roi_h <= 0:
            continue

        # Resize mask to ROI size
        resized_mask = cv2.resize(
            roi_mask.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # Prepare colored patch
        colored_patch = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        colored_patch[resized_mask] = color

        # Alpha blend only on masked pixels (single pass)
        roi_view = result[y1:y2, x1:x2]
        if roi_view.ndim == 2:
            # convert grayscale ROI to 3ch if needed
            roi_view = cv2.cvtColor(roi_view, cv2.COLOR_GRAY2BGR)

        # Blend on masked pixels without using cv2.addWeighted on flattened arrays
        fg = colored_patch[resized_mask].astype(np.float32)
        bg = roi_view[resized_mask].astype(np.float32)
        blended = (bg * (1.0 - alpha) + fg * alpha).astype(np.uint8)
        roi_view[resized_mask] = blended
        result[y1:y2, x1:x2] = roi_view

        if draw_roi:
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

    return result


def main():
    args = get_args()
    model_path = args.model
    score_threshold = args.score_threshold
    alpha = args.alpha
    draw_roi = args.draw_roi

    # Image Path and Define Regions of Interest (ROIs)
    # Optionally, you can use the results of an object detection model (e.g., YOLO) as ROIs
    image_path = "sample.jpg"
    rois_unnormalized = np.array(
        [
            [190, 0, 626, 533],
            [183, 239, 679, 497],
            [478, 0, 800, 532],
        ],
        dtype=np.float32,
    )

    # Load Model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_size = onnx_session.get_inputs()[0].shape
    input_height, input_width = input_size[2], input_size[3]

    # Read Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Preprocess
    input_image, rois = preprocess(image, rois_unnormalized, input_width, input_height)

    # Run Inference
    start_time = time.time()
    result = onnx_session.run(None, {"images": input_image, "rois": rois})
    masks = result[0]
    binary_masks = result[1][0][0]
    elapsed_time = time.time() - start_time
    print(f"Inference Time: {elapsed_time * 1000:.1f} ms")

    # Postprocess
    roi_masks = postprocess(masks, rois_unnormalized, score_threshold)

    binary_masks = cv2.resize(binary_masks, (image.shape[1], image.shape[0]))

    # Visualize
    debug_image = visualize(image, roi_masks, alpha, draw_roi)

    # Display Result
    cv2.imshow("RHIS Demo: binary_masks", binary_masks)
    cv2.imshow("RHIS Demo: masks", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
