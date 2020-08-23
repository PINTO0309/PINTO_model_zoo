### tensorflow==2.3.0

import tensorflow.compat.v1 as tf
import numpy as np
import cv2

def representative_dataset_gen():
  for count, image in enumerate(raw_test_data):
    print('image.shape:', count, image.shape)
    (m, s) = cv2.meanStdDev(image)
    m = m[0][0]
    s = s[0][0]
    image = image - m
    image = image / s if s>0 else image
    image = tf.image.resize(image, (128, 32))
    image = image[np.newaxis,:,:,:]
    yield [image]

# Full Integer Quantization - Input/Output=float32
raw_test_data = np.load('handwritten_dataset.npy', allow_pickle=True)
graph_def_file="simpleHTR_freeze_graph_opt.pb"
input_arrays=["input"]
output_arrays=['CTCGreedyDecoder','CTCGreedyDecoder:1','CTCGreedyDecoder:2','CTCGreedyDecoder:3']
input_tensor={"input":[1,128,32,1]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('simpleHTR_128x32_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - simpleHTR_128x32_full_integer_quant.tflite")
