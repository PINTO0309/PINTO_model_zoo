### tensorflow==2.3.1

import tensorflow as tf

# Weight Quantization - Input/Output=float32
height = 256
width  = 256
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
tflite_model = converter.convert()
with open('3dbox_mbnv2_{}x{}_float32.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Weight Quantization complete! - 3dbox_mbnv2_{}x{}_float32.tflite'.format(height, width))

height = 320
width  = 320
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
tflite_model = converter.convert()
with open('3dbox_mbnv2_{}x{}_float32.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Weight Quantization complete! - 3dbox_mbnv2_{}x{}_float32.tflite'.format(height, width))

'''
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_1'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 320, 320, 3)
        name: serving_default_input_1:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['confidence'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 2)
        name: StatefulPartitionedCall:0
    outputs['dimensions'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 3)
        name: StatefulPartitionedCall:1
    outputs['orientation'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 2, 2)
        name: StatefulPartitionedCall:2
  Method name is: tensorflow/serving/predict
'''