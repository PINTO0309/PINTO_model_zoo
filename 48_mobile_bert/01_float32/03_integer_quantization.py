### tensorflow-gpu==1.15.2

import tensorflow as tf
import numpy as np

def representative_dataset_gen():
  for i, data in enumerate(raw_test_data):
    input_ids = data[0]
    input_mask = data[1]
    segment_ids = data[2]
    print('Number of processed cases:', i)
    yield [input_ids, input_mask, segment_ids]

raw_test_data = np.load('mobilebert_dataset.npy', allow_pickle=True)

# Integer Quantization
converter = tf.lite.TFLiteConverter.from_saved_model('experiment/saved_model/1596242469')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new__converter = True
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('mobilebert_english_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - mobilebert_english_integer_quant.tflite")

