### tf-nightly-2.2.0-dev20200502

import tensorflow as tf
import numpy as np

def representative_dataset_gen():
    for image in raw_test_data:
        image = tf.image.resize(image, (256, 448))
        image = image[np.newaxis,:,:,:]
        image = image - 127.5
        image = image * 0.007843
        yield [image]

tf.compat.v1.enable_eager_execution()

raw_test_data = np.load('calibration_data_img.npy', allow_pickle=True)

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('human_pose_estimation_3d_0001_256x448_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - human_pose_estimation_3d_0001_256x448_integer_quant.tflite")