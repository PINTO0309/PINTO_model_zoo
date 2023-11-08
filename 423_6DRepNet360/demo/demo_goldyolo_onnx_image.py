#!/usr/bin/env python

import os
import copy
import cv2
from tqdm import tqdm
import glob
import numpy as np
import onnxruntime
from argparse import ArgumentParser
from typing import Tuple, Optional, List
from math import cos, sin


class GoldYOLOONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'gold_yolo_n_head_post_0277_0.5071_1x3x480x640.onnx',
        class_score_th: Optional[float] = 0.35,
        providers: Optional[List] = [
            # (
            #     'TensorrtExecutionProvider', {
            #         'trt_engine_cache_enable': True,
            #         'trt_engine_cache_path': '.',
            #         'trt_fp16_enable': True,
            #     }
            # ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        """GoldYOLOONNX

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for GoldYOLO

        class_score_th: Optional[float]
            Score threshold. Default: 0.35

        providers: Optional[List]
            Name of onnx execution providers
            Default:
            [
                (
                    'TensorrtExecutionProvider', {
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': '.',
                        'trt_fp16_enable': True,
                    }
                ),
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        """
        # Threshold
        self.class_score_th = class_score_th

        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs()
        ]


    def __call__(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """YOLOv7ONNX

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        boxes: np.ndarray
            Predicted boxes: [N, y1, x1, y2, x2]

        scores: np.ndarray
            Predicted box scores: [N, score]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = self.__preprocess(
            temp_image,
        )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        boxes = self.onnx_session.run(
            self.output_names,
            {input_name: inferece_image for input_name in self.input_names},
        )[0]

        # PostProcess
        result_boxes, result_scores = \
            self.__postprocess(
                image=temp_image,
                boxes=boxes,
            )

        return result_boxes, result_scores


    def __preprocess(
        self,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        """__preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Normalization + BGR->RGB
        resized_image = cv2.resize(
            image,
            (
                int(self.input_shapes[0][3]),
                int(self.input_shapes[0][2]),
            )
        )
        resized_image = np.divide(resized_image, 255.0)
        resized_image = resized_image[..., ::-1]
        resized_image = resized_image.transpose(swap)
        resized_image = np.ascontiguousarray(
            resized_image,
            dtype=np.float32,
        )
        return resized_image


    def __postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """__postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]

        Returns
        -------
        result_boxes: np.ndarray
            Predicted boxes: [N, y1, x1, y2, x2]

        result_scores: np.ndarray
            Predicted box confs: [N, score]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        """
        Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0

        batchno_classid_y1x1y2x2_score: float32[N,7]
        """
        result_boxes = []
        result_scores = []
        if len(boxes) > 0:
            scores = boxes[:, 6:7]
            keep_idxs = scores[:, 0] > self.class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                for box, score in zip(boxes_keep, scores_keep):
                    x_min = int(max(box[2], 0) * image_width / self.input_shapes[0][3])
                    y_min = int(max(box[3], 0) * image_height / self.input_shapes[0][2])
                    x_max = int(min(box[4], self.input_shapes[0][3]) * image_width / self.input_shapes[0][3])
                    y_max = int(min(box[5], self.input_shapes[0][2]) * image_height / self.input_shapes[0][2])

                    result_boxes.append(
                        [x_min, y_min, x_max, y_max]
                    )
                    result_scores.append(
                        score
                    )

        return np.asarray(result_boxes), np.asarray(result_scores)


def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)
    return img


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='gold_yolo_n_head_post_0277_0.5071_1x3x480x640.onnx',
    )
    parser.add_argument(
        '-i',
        '--images_path',
        type=str,
        default="./images",
    )
    parser.add_argument(
        '-o',
        '--output_path',
        type=str,
        default="./output",
    )
    args = parser.parse_args()

    model = GoldYOLOONNX(
        model_path=args.model,
    )

    files = sorted(glob.glob(f"{args.images_path}/*.png") + glob.glob(f"{args.images_path}/*.jpg"))
    os.makedirs(args.output_path, exist_ok=True)

    session_option = onnxruntime.SessionOptions()
    session_option.log_severity_level = 3
    onnx_session = onnxruntime.InferenceSession(
        'sixdrepnet360_Nx3x224x224.onnx',
        sess_options=session_option,
        providers=[
            # (
            #     'TensorrtExecutionProvider', {
            #         'trt_engine_cache_enable': True,
            #         'trt_engine_cache_path': '.',
            #         'trt_fp16_enable': True,
            #     }
            # ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    )

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    for file in tqdm(files, dynamic_ncols=True):
        image =  cv2.imread(file)
        debug_image = copy.deepcopy(image)
        boxes, scores = model(debug_image)

        image_height = debug_image.shape[0]
        image_width = debug_image.shape[1]

        for box, score in zip(boxes, scores):
            x1: int = box[0]
            y1: int = box[1]
            x2: int = box[2]
            y2: int = box[3]

            cx: int = (x1 + x2) // 2
            cy: int = (y1 + y2) // 2
            w: int = abs(x2 - x1)
            h: int = abs(y2 - y1)
            ew: float = w * 1.2
            eh: float = h * 1.2
            ex1 = int(cx - ew / 2)
            ex2 = int(cx + ew / 2)
            ey1 = int(cy - eh / 2)
            ey2 = int(cy + eh / 2)

            ex1 = ex1 if ex1 >= 0 else 0
            ex2 = ex2 if ex2 <= image_width else image_width
            ey1 = ey1 if ey1 >= 0 else 0
            ey2 = ey2 if ey2 <= image_height else image_height

            inference_image = copy.deepcopy(debug_image)
            head_image_bgr = inference_image[ey1:ey2, ex1:ex2, :]
            resized_image_bgr = cv2.resize(head_image_bgr, (256, 256))
            cropped_image_bgr = resized_image_bgr[16:240, 16:240, :]

            # inference
            cropped_image_rgb: np.ndarray = cropped_image_bgr[..., ::-1]
            normalized_image_rgb: np.ndarray = (cropped_image_rgb / 255.0 - mean) / std
            normalized_image_rgb = normalized_image_rgb.transpose(2,0,1)
            normalized_image_rgb: np.ndarray = normalized_image_rgb[np.newaxis, ...]
            normalized_image_rgb: np.ndarray = normalized_image_rgb.astype(np.float32)
            yaw_pitch_roll: np.ndarray = \
                onnx_session.run(
                    None,
                    {'input': normalized_image_rgb},
                )[0]
            yaw_deg = yaw_pitch_roll[0][0]
            pitch_deg = yaw_pitch_roll[0][1]
            roll_deg = yaw_pitch_roll[0][2]
            draw_axis(debug_image, yaw_deg, pitch_deg, roll_deg, tdx=float(cx), tdy=float(cy), size=100)

            cv2.rectangle(
                debug_image,
                (box[0], box[1]),
                (box[2], box[3]),
                (255,255,255),
                2,
            )
            cv2.rectangle(
                debug_image,
                (box[0], box[1]),
                (box[2], box[3]),
                (0,0,255),
                1,
            )
            cv2.putText(
                debug_image,
                f'{score[0]:.2f}',
                (
                    box[0],
                    box[1]-10 if box[1]-10 > 0 else 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f'{score[0]:.2f}',
                (
                    box[0],
                    box[1]-10 if box[1]-10 > 0 else 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(f'{args.output_path}/{os.path.basename(file)}', debug_image)
        cv2.imshow("test", debug_image)

        key = cv2.waitKey(0)
        if key == 27: # ESC
            break



if __name__ == "__main__":
    main()
