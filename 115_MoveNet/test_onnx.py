import onnx
import onnxruntime
import numpy as np
import cv2
import pprint

model_path = "model_float32.onnx"

def main():
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    sess = onnxruntime.InferenceSession(model_path)

    image = cv2.imread('test.png')
    frame = cv2.resize(image, (256, 256))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0)
    frame = frame.astype('float32')

    inputs = {sess.get_inputs()[0].name: frame}
    outputs = sess.run(None, inputs)

    pprint.pprint(outputs)

if __name__ == "__main__":
    main()