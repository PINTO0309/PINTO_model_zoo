### tf-nightly==2.2.0-dev20200414

import tensorflow as tf
import numpy as np

## Generating a calibration data set
def representative_dataset_gen():
    raw_test_data = np.load('calibration_data_img.npy')
    for data in raw_test_data:
        calibration_data = data[np.newaxis, :, :, :].astype(np.float32)
        calibration_data = (calibration_data - 0.45) / 0.225
        yield [calibration_data]

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('monodepth2_integer_quant.tflite', 'wb') as w:
   w.write(tflite_quant_model)
print("Integer Quantization complete! - monodepth2_integer_quant.tflite")

