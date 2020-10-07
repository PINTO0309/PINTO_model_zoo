### tensorflow==2.3.1

import tensorflow as tf

# Weight Quantization - Input/Output=float32
height = 480
width  = 640
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_nyu_{}x{}'.format(height, width))
tflite_model = converter.convert()
with open('dense_depth_nyu_{}x{}_float32.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('tflite convert complete! - dense_depth_nyu_{}x{}_float32.tflite'.format(height, width))

'''
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_1'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 480, 640, 3)
        name: serving_default_input_1:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_1'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 240, 320, 1)
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict
'''