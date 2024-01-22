#!/usr/bin/env python

import copy
import cv2
import time
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Tuple, Optional, List, Dict

BODY_COLORS = [
    (110, 64, 170), (143, 61, 178), (178, 60, 178), (210, 62, 167),
    (238, 67, 149), (255, 78, 125), (255, 94, 99),  (255, 115, 75),
    (255, 140, 56), (239, 167, 47), (217, 194, 49), (194, 219, 64),
    (175, 240, 91), (135, 245, 87), (96, 247, 96),  (64, 243, 115),
    (40, 234, 141), (28, 219, 169), (26, 199, 194), (33, 176, 213),
    (47, 150, 224), (65, 125, 224), (84, 101, 214), (99, 81, 195)
]

@dataclass(frozen=False)
class Box():
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _class_score_th: float = 0.35
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    # onnx/tflite
    _interpreter = None
    _inference_model = None
    _providers = None
    _swap = (2, 0, 1)
    _h_index = 2
    _w_index = 3

    # onnx
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    # tflite
    _input_details = None
    _output_details = None

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = '',
        class_score_th: Optional[float] = 0.35,
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        self._runtime = runtime
        self._model_path = model_path
        self._class_score_th = class_score_th
        self._providers = providers

        # Model loading
        if self._runtime == 'onnx':
            import onnxruntime # type: ignore
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self._interpreter = \
                onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_option,
                    providers=providers,
                )
            self._providers = self._interpreter.get_providers()
            self._input_shapes = [
                input.shape for input in self._interpreter.get_inputs()
            ]
            self._input_names = [
                input.name for input in self._interpreter.get_inputs()
            ]
            self._input_dtypes = [
                self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter.get_inputs()
            ]
            self._output_shapes = [
                output.shape for output in self._interpreter.get_outputs()
            ]
            self._output_names = [
                output.name for output in self._interpreter.get_outputs()
            ]
            self._model = self._interpreter.run
            self._swap = (2, 0, 1)
            self._h_index = 2
            self._w_index = 3
            self.strides: int = 0

        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            if self._runtime == 'tflite_runtime':
                from tflite_runtime.interpreter import Interpreter # type: ignore
                self._interpreter = Interpreter(model_path=model_path)
            elif self._runtime == 'tensorflow':
                import tensorflow as tf # type: ignore
                self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_shapes = [
                input.get('shape', None) for input in self._input_details
            ]
            self._input_names = [
                input.get('name', None) for input in self._input_details
            ]
            self._input_dtypes = [
                input.get('dtype', None) for input in self._input_details
            ]
            self._output_shapes = [
                output.get('shape', None) for output in self._output_details
            ]
            self._output_names = [
                output.get('name', None) for output in self._output_details
            ]
            self._model = self._interpreter.get_signature_runner()
            self._swap = (0, 1, 2)
            self._h_index = 1
            self._w_index = 2

    @abstractmethod
    def __call__(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f'{input_name}': input_data \
                for input_name, input_data in zip(self._input_names, input_datas)
        }
        if self._runtime == 'onnx':
            outputs = [
                output for output in \
                    self._model(
                        output_names=self._output_names,
                        input_feed=datas,
                    )
            ]
            return outputs
        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            outputs = [
                output for output in \
                    self._model(
                        **datas
                    ).values()
            ]
            return outputs

    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        pass

class YOLOX(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'yolox_x_body_head_hand_face_0076_0.5228_post_1x3x480x640.onnx',
        class_score_th: Optional[float] = 0.35,
        providers: Optional[List] = None,
    ):
        """YOLOX

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for YOLOX. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for YOLOX

        class_score_th: Optional[float]
            Score threshold. Default: 0.35

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            class_score_th=class_score_th,
            providers=providers,
        )

    def __call__(
        self,
        image: np.ndarray,
    ) -> List[Box]:
        """YOLOX

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [N, classid, score, x1, y1, x2, y2]
        """
        temp_image = copy.deepcopy(image)
        # PreProcess
        resized_image = \
            self._preprocess(
                temp_image,
            )
        # Inference
        inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inferece_image])
        boxes = outputs[0]
        # PostProcess
        result_boxes = \
            self._postprocess(
                image=temp_image,
                boxes=boxes,
            )
        return result_boxes

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Resize + Transpose
        resized_image = cv2.resize(
            image,
            (
                int(self._input_shapes[0][self._w_index]),
                int(self._input_shapes[0][self._h_index]),
            )
        )
        resized_image = resized_image.transpose(self._swap)
        resized_image = \
            np.ascontiguousarray(
                resized_image,
                dtype=np.float32,
            )
        return resized_image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2]
        """

        """
        Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0

        batchno_classid_score_x1y1x2y2: float32[N,7]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes: List[Box] = []

        if len(boxes) > 0:
            scores = boxes[:, 2:3]
            keep_idxs = scores[:, 0] > self._class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                for box, score in zip(boxes_keep, scores_keep):
                    x_min = int(max(0, box[3]) * image_width / self._input_shapes[0][self._w_index])
                    y_min = int(max(0, box[4]) * image_height / self._input_shapes[0][self._h_index])
                    x_max = int(min(box[5], self._input_shapes[0][self._w_index]) * image_width / self._input_shapes[0][self._w_index])
                    y_max = int(min(box[6], self._input_shapes[0][self._h_index]) * image_height / self._input_shapes[0][self._h_index])
                    result_boxes.append(
                        Box(
                            classid=int(box[1]),
                            score=float(score),
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                        )
                    )
        return result_boxes

class BodyPix(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'bodypix_resnet50_stride16_Nx3x384x288.onnx',
        providers: Optional[List] = None,
    ):
        """BodyPix

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for BodyPix. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for BodyPix

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            providers=providers,
        )
        self._swap = (0,3,1,2)
        self._mean = np.asarray([0.0, 0.0, 0.0])
        self._std = np.asarray([1.0, 1.0, 1.0])
        import onnx
        model_proto = onnx.load(f=model_path)
        float_segments_raw_output = [v for v in model_proto.graph.value_info if v.name == 'float_segments_raw_output']
        if len(float_segments_raw_output) >= 1:
            w = float_segments_raw_output[0].type.tensor_type.shape.dim[-1].dim_value
            self.strides = self._input_shapes[0][self._w_index] // w

    def __call__(
        self,
        image: np.ndarray,
        body_boxes: List[Box]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        foreground_mask_zero_or_255: List[np.ndarray]
            Predicted foreground mask: [batch, H, W, 3]. 0 or 255

        colored_mask_classid: List[np.ndarray]
            Predicted colored mask: [batch, H, W, 1]. 0 - 24

        keypoints_classidscorexy: np.ndarray
            Predicted keypoints: [batch, N, 4]. classid, score, x, y
        """
        temp_image = copy.deepcopy(image)
        # PreProcess
        resized_images = \
            self._preprocess(
                image=temp_image,
                body_boxes=body_boxes,
            )

        result_foreground_mask_zero_or_255s: np.ndarray = np.asarray([], dtype=np.float32)
        result_colored_mask_classids: np.ndarray = np.asarray([], dtype=np.float32)
        result_keypoints_classidscorexys: np.ndarray = np.asarray([], dtype=np.float32)

        if len(resized_images) > 0:
            # Inference
            outputs = super().__call__(input_datas=[resized_images])
            foreground_mask_zero_or_255s = outputs[0]
            colored_mask_classids = outputs[1]
            keypoints_classidscorexys = outputs[2]
            # PostProcess
            result_foreground_mask_zero_or_255s, result_colored_mask_classids, result_keypoints_classidscorexys = \
                self._postprocess(
                    body_boxes=body_boxes,
                    foreground_mask_zero_or_255s=foreground_mask_zero_or_255s,
                    colored_mask_classids=colored_mask_classids,
                    keypoints_classidscorexys=keypoints_classidscorexys,
                )
        return result_foreground_mask_zero_or_255s, result_colored_mask_classids, result_keypoints_classidscorexys

    def _preprocess(
        self,
        image: np.ndarray,
        body_boxes: List[Box],
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        body_images: List[np.ndarray] = []
        body_images_np: np.ndarray = np.asarray([], dtype=np.float32)
        if len(body_boxes) > 0:
            for body in body_boxes:
                body_image: np.ndarray = image[body.y1:body.y2, body.x1:body.x2, :]
                resized_body_image = cv2.resize(
                    body_image,
                    (
                        int(self._input_shapes[0][self._w_index]),
                        int(self._input_shapes[0][self._h_index]),
                    )
                )
                resized_body_image = resized_body_image[..., ::-1]
                body_images.append(resized_body_image)
            body_images_np = np.asarray(body_images, dtype=np.float32)
            body_images_np = body_images_np.transpose(self._swap)
            body_images_np = body_images_np.astype(self._input_dtypes[0])
        return body_images_np

    def _postprocess(
        self,
        body_boxes: List[Box],
        foreground_mask_zero_or_255s: np.ndarray,
        colored_mask_classids: np.ndarray,
        keypoints_classidscorexys: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """_postprocess

        Parameters
        ----------
        foreground_mask_zero_or_255: np.ndarray
            Predicted foreground mask: [batch, 3, H, W]. 0 or 255

        colored_mask_classid: np.ndarray
            Predicted colored mask: [batch, 1, H, W]. 0 - 24

        keypoints_classidscorexy: np.ndarray
            Predicted keypoints: [batch, N, 4]. classid, score, x, y

        Returns
        -------
        foreground_mask_zero_or_255: List[np.ndarray]
            Predicted foreground mask: [batch, H, W, 3]. 0 or 255

        colored_mask_classid: List[np.ndarray]
            Predicted colored mask: [batch, H, W, 1]. 0 - 24

        keypoints_classidscorexy: np.ndarray
            Predicted keypoints: [batch, N, 4]. classid, score, x, y
        """
        rescaled_foreground_mask_zero_or_255s: List[np.ndarray] = []
        rescaled_colored_mask_classids: List[np.ndarray] = []
        rescaled_keypoints_classidscorexys = copy.deepcopy(keypoints_classidscorexys)

        if len(foreground_mask_zero_or_255s) > 0:
            for foreground_mask_zero_or_255, colored_mask_classid, rescaled_keypoints_classidscorexy, body_box \
                in zip(foreground_mask_zero_or_255s, colored_mask_classids, rescaled_keypoints_classidscorexys, body_boxes):

                inf_h = foreground_mask_zero_or_255.shape[1]
                inf_w = foreground_mask_zero_or_255.shape[2]
                w = abs(body_box.x2 - body_box.x1)
                h = abs(body_box.y2 - body_box.y1)

                foreground_mask_zero_or_255: np.ndarray = cv2.resize(foreground_mask_zero_or_255.transpose(1,2,0).astype(np.uint8), (w, h)) # [H, W, 3]
                rescaled_foreground_mask_zero_or_255s.append(foreground_mask_zero_or_255)

                colored_mask_classid: np.ndarray = cv2.resize(colored_mask_classid.transpose(1,2,0).astype(np.uint8), (w, h)) # [H, W, 1]
                rescaled_colored_mask_classids.append(colored_mask_classid[..., np.newaxis])

                rescaled_keypoints_classidscorexy[..., 2] = rescaled_keypoints_classidscorexy[..., 2] / (inf_w / w) + body_box.x1
                rescaled_keypoints_classidscorexy[..., 3] = rescaled_keypoints_classidscorexy[..., 3] / (inf_h / h) + body_box.y1

        return rescaled_foreground_mask_zero_or_255s, rescaled_colored_mask_classids, rescaled_keypoints_classidscorexys

def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
):
    """Function to draw a dashed line"""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = [int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)]
        end = [int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)]
        cv2.line(image, tuple(start), tuple(end), color, thickness)

def draw_dashed_rectangle(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10
):
    """Function to draw a dashed rectangle"""
    tl_tr = (bottom_right[0], top_left[1])
    bl_br = (top_left[0], bottom_right[1])
    draw_dashed_line(image, top_left, tl_tr, color, thickness, dash_length)
    draw_dashed_line(image, tl_tr, bottom_right, color, thickness, dash_length)
    draw_dashed_line(image, bottom_right, bl_br, color, thickness, dash_length)
    draw_dashed_line(image, bl_br, top_left, color, thickness, dash_length)

def extract_max_score_points_unique(points: np.ndarray, radius: int=20):
    """
    Extracts the points with the maximum score within a specified radius for each classid, ensuring uniqueness.

    Parameters:
    points (np.ndarray): An array of shape [n, 4] where each row represents [classid, score, x, y].
    radius (float): The radius within which to search for points.

    Returns:
    np.ndarray: An array of extracted unique points with the maximum score within the specified radius for each classid.
    """
    # Initialize an empty list to store the maximum score points
    max_score_points = []
    # Iterate over each unique classid in the array
    for classid in np.unique(points[:, 0]):
        class_points = points[points[:, 0] == classid]
        while class_points.size > 0:
            # Take the first point as the reference
            reference_point = class_points[0]
            _, _, ref_x, ref_y = reference_point
            # Calculate the distance from the reference point to all other points in the same class
            distances = np.sqrt((class_points[:, 2] - ref_x)**2 + (class_points[:, 3] - ref_y)**2)
            # Filter points within the specified radius
            within_radius = class_points[distances <= radius]
            # Find the point with the maximum score within the radius
            if within_radius.size > 0:
                max_score_point = within_radius[within_radius[:, 1].argmax()]
                max_score_points.append(max_score_point)
            # Remove the selected points from the class_points
            class_points = np.array([point for point in class_points if list(point) not in within_radius.tolist()])
    # Convert the list of points back to a NumPy array
    return np.array(max_score_points)

def affine_transform(image: np.ndarray, width: int, height: int, dx:int, dy:int):
    # Create a transformation matrix for a parallel shift
    afin_matrix = np.float32([[1,0,dx],[0,1,dy]])
    # apply an infinite transformation
    afin_image = \
        cv2.warpAffine(
            src=image,
            M=afin_matrix,
            dsize=(width, height)
        )
    return afin_image

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-dm',
        '--detection_model',
        type=str,
        default='yolox_x_body_head_hand_face_0076_0.5228_post_1x3x480x640.onnx',
    )
    parser.add_argument(
        '-bm',
        '--bodypix_model',
        type=str,
        default='bodypix_resnet50_stride16_Nx3x384x288.onnx',
    )
    parser.add_argument(
        '-v',
        '--video',
        type=str,
        default="0",
    )
    parser.add_argument(
        '-ep',
        '--execution_provider',
        type=str,
        choices=['cpu', 'cuda', 'tensorrt'],
        default='tensorrt',
    )
    args = parser.parse_args()

    providers: List[Tuple[str, Dict] | str] = None
    if args.execution_provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif args.execution_provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif args.execution_provider == 'tensorrt':
        providers = [
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

    model_yolox = \
        YOLOX(
            model_path=args.detection_model,
            providers=providers,
        )
    model_bodypix = \
        BodyPix(
            model_path=args.bodypix_model,
            providers=providers,
        )

    cap = cv2.VideoCapture(
        int(args.video) if is_parsable_to_int(args.video) else args.video
    )
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(w, h*2),
    )

    while cap.isOpened():
        res, image = cap.read()
        if not res:
            break

        debug_image = copy.deepcopy(image)
        inference_image = copy.deepcopy(image)
        mask_image = copy.deepcopy(image)

        start_time = time.perf_counter()

        # Body Detection
        boxes = model_yolox(inference_image)

        body_boxes: List[Box] = []
        for box in boxes:
            classid: int = box.classid
            color = (255,255,255)
            if classid == 0:
                color = (255,0,0)
            elif classid == 1:
                color = (0,0,255)
            elif classid == 2:
                color = (0,255,0)
            elif classid == 3:
                color = (0,200,255)

            # Make Face box
            if classid == 0:
                body_boxes.append(box)

            if classid != 3:
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), 2)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 1)
            else:
                draw_dashed_rectangle(
                    image=debug_image,
                    top_left=(box.x1, box.y1),
                    bottom_right=(box.x2, box.y2),
                    color=color,
                    thickness=2,
                    dash_length=10
                )
                face_image = copy.deepcopy(debug_image[box.y1:box.y2, box.x1:box.x2, :])
                if w//50 > 0 and h//50 > 0:
                    debug_image[box.y1:box.y2, box.x1:box.x2, :] = cv2.resize(cv2.resize(face_image, (w//50, h//50)), (box.x2-box.x1, box.y2-box.y1))

        # foreground_mask_zero_or_255: [H, W, 3], 0 or 255
        # colored_mask_classid: [H, W, 1], 0 - 24
        # keypoints_classidscorexy: [N, 4] [keypoint_classid, score, x, y]
        foreground_mask_zero_or_255s, colored_mask_classids, keypoints_classidscorexys = model_bodypix(inference_image, body_boxes)

        remap_foreground_mask_zero_or_255 = np.zeros_like(mask_image, dtype=np.uint8)
        remap_colored_mask_classid = np.zeros_like(mask_image, dtype=np.uint8)[..., 0:1]
        for foreground_mask_zero_or_255, colored_mask_classid, keypoints_classidscorexy, body_box \
            in zip(foreground_mask_zero_or_255s, colored_mask_classids, keypoints_classidscorexys, body_boxes):

            # Fine-tune position of mask image
            number_of_fine_tuning_pixels: int = model_bodypix.strides // 2
            if number_of_fine_tuning_pixels > 0:
                h = abs(body_box.y2 - body_box.y1)
                w = abs(body_box.x2 - body_box.x1)
                foreground_mask_zero_or_255 = \
                    affine_transform(
                        image=foreground_mask_zero_or_255,
                        height=h,
                        width=w,
                        dx=-number_of_fine_tuning_pixels / (model_bodypix._input_shapes[0][3] / w),
                        dy=-number_of_fine_tuning_pixels / (model_bodypix._input_shapes[0][2] / h),
                    )
                colored_mask_classid = \
                    affine_transform(
                        image=colored_mask_classid,
                        height=h,
                        width=w,
                        dx=-number_of_fine_tuning_pixels / (model_bodypix._input_shapes[0][3] / w),
                        dy=-number_of_fine_tuning_pixels / (model_bodypix._input_shapes[0][2] / h),
                    )[..., np.newaxis]

            # Eliminate low score keypoints
            score_keep = keypoints_classidscorexy[..., 1] >= 0.85
            keypoints_classidscorexy = keypoints_classidscorexy[score_keep, :]

            # Eliminate duplicate detection of neighboring keypoints
            if len(keypoints_classidscorexy) > 0:
                keypoints_classidscorexy = extract_max_score_points_unique(keypoints_classidscorexy)

            _ = [
                cv2.circle(debug_image, (int(landmark[2]), int(landmark[3])), 1, (0, 255, 0), 2) for landmark in keypoints_classidscorexy
            ]

            # binary mask
            remap_foreground_mask_zero_or_255[body_box.y1:body_box.y2, body_box.x1:body_box.x2, :] += foreground_mask_zero_or_255

            # colored mask
            remap_colored_mask_classid[body_box.y1:body_box.y2, body_box.x1:body_box.x2, :] = colored_mask_classid
            part_colors = np.asarray(BODY_COLORS, dtype=np.uint8)
            colored_mask = part_colors[remap_colored_mask_classid[..., 0]]

        remap_foreground_mask_zero_or_255 = np.clip(remap_foreground_mask_zero_or_255, 0, 255, dtype=np.uint8)
        mask_image = cv2.bitwise_and(mask_image, remap_foreground_mask_zero_or_255)
        colored_mask = cv2.bitwise_and(colored_mask, remap_foreground_mask_zero_or_255)

        elapsed_time = time.perf_counter() - start_time

        cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 150, 196), 1, cv2.LINE_AA)

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break

        cv2.imshow("test", debug_image)
        cv2.imshow("mask", mask_image)
        cv2.imshow("colored_mask", colored_mask)
        video_writer.write(np.vstack((debug_image, colored_mask)))

    if video_writer:
        video_writer.release()

    if cap:
        cap.release()

if __name__ == "__main__":
    main()
