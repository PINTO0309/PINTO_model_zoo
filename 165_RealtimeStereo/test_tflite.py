from tensorflow.lite.python.interpreter import Interpreter

import cv2
import numpy as np
import time


class RealtimeStereo():
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        self.interpreter = Interpreter(model_path, num_threads=4)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        input_shape = self.input_details[0]['shape']
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.channels = input_shape[2]

        self.output_details = self.interpreter.get_output_details()
        self.output_shape = self.output_details[0]['shape']

    def preprocess(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(
            img,
            (self.input_width,self.input_height)
        ).astype(np.float32)

        # Scale input pixel values to -1 to 1
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        img_input = ((img_input/ 255.0 - mean) / std)
        # img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis,:,:,:]

        return img_input.astype(np.float32)

    def run(self, left, right):
        input_left = self.preprocess(left)
        input_right = self.preprocess(right)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_left)
        self.interpreter.set_tensor(self.input_details[1]['index'], input_right)
        self.interpreter.invoke()

        disparity = self.interpreter.get_tensor(self.output_details[0]['index'])

        return np.squeeze(disparity)

if __name__ == '__main__':
    # model_path = 'saved_model/model_float32.tflite'
    model_path = 'saved_model/model_float16_quant.tflite'
    # model_path = 'saved_model/model_dynamic_range_quant.tflite'
    realtimeStereo = RealtimeStereo(model_path)

    img_left = cv2.imread('im0.png')
    img_right = cv2.imread('im1.png')

    start = time.time()

    disp = realtimeStereo.run(img_left, img_right)

    disp = cv2.resize(
        disp,
        (img_left.shape[1], img_left.shape[0]),
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    img = (disp*256).astype('uint16')
    cv2.imshow('disp', img)

    d_min = np.min(disp)
    d_max = np.max(disp)
    depth_map = (disp - d_min) / (d_max - d_min)
    depth_map = depth_map * 255.0
    depth_map = np.asarray(depth_map, dtype="uint8")
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

    end = time.time()
    eslapse = end - start
    print("depthmap : {}s".format(eslapse))

    cv2.imwrite('result.jpg', depth_map)

    cv2.imshow('output', depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()