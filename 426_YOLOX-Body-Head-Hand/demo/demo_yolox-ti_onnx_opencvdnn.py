"""
code cited from: https://qiita.com/UnaNancyOwen/items/650d79c88a58a3cc30ce
"""
import cv2
import onnx
import time
import numpy as np
from typing import List

# params
# WEIGHTS = "yolox_ti_body_head_hand_n_1x3x128x160_uint8.onnx"
# WEIGHTS = "yolox_ti_body_head_hand_n_1x3x256x320_uint8.onnx"
WEIGHTS = "yolox_ti_body_head_hand_n_1x3x480x640_uint8.onnx"
NUM_CLASSES = 3
SCORE_THRESHOLD = 0.60
IOU_THRESHOLD = 0.4
CAP_WIDTH = 640
CAP_HEIGHT = 480

# detection model class for yolox
class DetectionModel:
    # constructor
    def __init__(
        self,
        *,
        weight: str,
    ):
        self.__initialize(weight=weight)

    # set preferable backend
    def _setPreferableBackend(self, backend):
        self._interpreter.setPreferableBackend(backend)

    # set preferable target
    def _setPreferableTarget(self, target):
        self._interpreter.setPreferableTarget(target)

    # initialize
    def __initialize(
        self,
        *,
        weight: str,
    ):
        onnx_model = onnx.load(f=weight)
        self._input_shapes = [
            [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        ]
        self._input_names = [
            input.name for input in onnx_model.graph.input
        ]
        self._output_shapes = [
            [dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]
        ]
        self._output_names = [
            output.name for output in onnx_model.graph.output
        ]
        del onnx_model

        self._interpreter = cv2.dnn.readNet(weight)
        self._setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._model = self._interpreter
        self._h_index = 2
        self._w_index = 3
        strides = [8, 16, 32]
        self.grids, self.expanded_strides = \
            self.__create_grids_and_expanded_strides(strides=strides)

    # create grids and expanded strides
    def __create_grids_and_expanded_strides(
        self,
        *,
        strides: List[int],
    ):
        grids = []
        expanded_strides = []

        hsizes = [self._input_shapes[0][self._h_index] // stride for stride in strides]
        wsizes = [self._input_shapes[0][self._w_index] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)

        return grids, expanded_strides

    # detect objects
    def __call__(
        self,
        *,
        image: np.ndarray,
        score_threshold: float,
        iou_threshold: float,
    ):
        self.image_shape = image.shape
        input_blob, resize_ratio_w, resize_ratio_h = self.__preprocess(image=image)
        self._model.setInput(input_blob)
        output_layer = self._model.getUnconnectedOutLayersNames()[0] # "output"
        outputs = self._model.forward(output_layer)

        boxes, scores, class_ids = \
            self.__postprocess(
                output_blob=outputs,
                resize_ratio_w=resize_ratio_w,
                resize_ratio_h=resize_ratio_h,
            )
        boxes, scores, class_ids = \
            self.__nms(
                boxes=boxes,
                scores=scores,
                class_ids=class_ids,
                score_threshold=score_threshold,
                iou_threshold=iou_threshold,
            )
        return class_ids, scores, boxes

    # preprocess
    def __preprocess(
        self,
        *,
        image: np.ndarray,
    ):
        resize_ratio_w = self._input_shapes[0][self._w_index] / self.image_shape[1]
        resize_ratio_h = self._input_shapes[0][self._h_index] / self.image_shape[0]
        resized_image = \
            cv2.resize(
                image,
                dsize=(self._input_shapes[0][self._w_index], self._input_shapes[0][self._h_index])
            )
        input_blob = \
            cv2.dnn.blobFromImage(
                image=resized_image,
                scalefactor=1.0,
                size=(self._input_shapes[0][self._w_index], self._input_shapes[0][self._h_index]),
                mean=(0.0, 0.0, 0.0),
                swapRB=False,
                crop=False,
            )
        return input_blob, resize_ratio_w, resize_ratio_h

    # postprocess
    def __postprocess(
        self,
        *,
        output_blob: np.ndarray,
        resize_ratio_w: float,
        resize_ratio_h: float,
    ):
        output_blob[..., :2] = (output_blob[..., :2] + self.grids) * self.expanded_strides
        output_blob[..., 2:4] = np.exp(output_blob[..., 2:4]) * self.expanded_strides

        predictions: np.ndarray = output_blob[0]
        boxes = predictions[:, :4]
        boxes_xywh = np.ones_like(boxes)

        # yolox-ti
        boxes[:, 0] = boxes[:, 0] / resize_ratio_w
        boxes[:, 1] = boxes[:, 1] / resize_ratio_h
        boxes[:, 2] = boxes[:, 2] / resize_ratio_w
        boxes[:, 3] = boxes[:, 3] / resize_ratio_h
        boxes_xywh[:, 0] = (boxes[:, 0] - boxes[:, 2] * 0.5)
        boxes_xywh[:, 1] = (boxes[:, 1] - boxes[:, 3] * 0.5)
        boxes_xywh[:, 2] = ((boxes[:, 0] + boxes[:, 2] * 0.5) - boxes_xywh[:, 0])
        boxes_xywh[:, 3] = ((boxes[:, 1] + boxes[:, 3] * 0.5) - boxes_xywh[:, 1])

        scores = predictions[:, 4:5] * predictions[:, 5:]
        class_ids = scores.argmax(1)
        scores = scores[np.arange(len(class_ids)), class_ids]

        return boxes_xywh, scores, class_ids

    # non maximum suppression
    def __nms(
        self,
        *,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        score_threshold: float,
        iou_threshold: float,
    ):
        indices = \
            cv2.dnn.NMSBoxesBatched(
                bboxes=boxes,
                scores=scores,
                class_ids=class_ids,
                score_threshold=score_threshold,
                nms_threshold=iou_threshold,
            ) # OpenCV 4.7.0 or later

        keep_boxes = []
        keep_scores = []
        keep_class_ids = []
        for index in indices:
            keep_boxes.append(boxes[index])
            keep_scores.append(scores[index])
            keep_class_ids.append(class_ids[index])

        if len(keep_boxes) > 0:
            keep_boxes = np.vectorize(int)(keep_boxes)

        return keep_boxes, keep_scores, keep_class_ids

# get raudom colors
def get_colors(num: int):
    colors = []
    np.random.seed(0)
    for _ in range(num):
        color = np.random.randint(0, 256, [3]).astype(np.uint8)
        colors.append(color.tolist())
    return colors

# main
def main():
    # read image
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    # create detection model class for yolox
    model = DetectionModel(weight=WEIGHTS)

    # detect objects
    score_threshold = SCORE_THRESHOLD
    iou_threshold = IOU_THRESHOLD

    while cap.isOpened():
        res, image = cap.read()
        if not res:
            break

        start_time = time.perf_counter()
        class_ids, scores, boxes = \
            model(
                image=image,
                score_threshold=score_threshold,
                iou_threshold=iou_threshold,
            )
        elapsed_time = time.perf_counter() - start_time
        cv2.putText(
            image,
            f'{elapsed_time*1000:.2f} ms',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f'{elapsed_time*1000:.2f} ms',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        # draw objects
        num_classes = NUM_CLASSES
        colors = get_colors(num_classes)
        for box, score, class_id in zip(boxes, scores, class_ids):
            color = colors[class_id]
            thickness = 2
            line_type = cv2.LINE_AA
            cv2.rectangle(image, box, color, thickness, line_type)

        # show image
        cv2.imshow("image", image)
        key = cv2.waitKey(1)
        if key == 27: # ESC
            break

    if cap is not None:
        cap.release()

if __name__ == '__main__':
    main()