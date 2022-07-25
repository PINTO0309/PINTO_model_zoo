import os
import onnxruntime as ort
import numpy as np
import cv2
import time
import argparse


class ExposeCorrection:
    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        model_inputs = self.session.get_inputs()

        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        input_shape = model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        output_shape = model_outputs[0].shape
        output_height = output_shape[2]
        output_width = output_shape[3]

    def preprocess(self, img_bgr):
        img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_input, (self.input_width, self.input_height))
        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis, :, :, :]
        img_input = img_input.astype(np.float32) / 255.0
        return img_input

    def postprocess(self, output):
        out = output[0][0].transpose(1, 2, 0)
        out = (out * 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out

    def inference(self, img_bgr):
        img_input = self.preprocess(img_bgr)
        outputs = self.session.run(self.output_names,
                                   {self.input_names[0]: img_input})
        out = self.postprocess(outputs)
        return out

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--video", type=str, default="/dev/video0")
    args = parser.parse_args()

    model_path = args.model
    video_path = args.video

    exp = ExposeCorrection(model_path)

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        res, img = cap.read()
        if res is False:
            break

        t = time.time()
        img_output = exp.inference(img)
        dt = time.time() - t
        print(f"{1 / dt:.4f} FSP")

        resized = cv2.resize(img, img_output.shape[:2][::-1])
        shown = np.hstack([resized, img_output])
        cv2.imshow("IAT exposure correction", shown)

        key = cv2.waitKey(10)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
