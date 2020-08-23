import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="dstbpp512fl_sythtext_512x512_integer_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print('input:', input_details)
print('')
print('output:', output_details)

input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data1 = interpreter.get_tensor(output_details[0]['index'])
output_data2 = interpreter.get_tensor(output_details[1]['index'])
output_data3 = interpreter.get_tensor(output_details[2]['index'])
output_data4 = interpreter.get_tensor(output_details[3]['index'])
print('output_data1.shape:', output_data1.shape)
print('output_data2.shape:', output_data2.shape)
print('output_data3.shape:', output_data3.shape)
print('output_data4.shape:', output_data4.shape)