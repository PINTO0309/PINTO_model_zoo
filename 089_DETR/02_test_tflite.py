import numpy as np
import pprint
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

if __name__ == '__main__':

    # Tensorflow Lite
    interpreter = Interpreter(model_path='model_float32.tflite', num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data1 = interpreter.get_tensor(output_details[0]['index'])
    output_data2 = interpreter.get_tensor(output_details[1]['index'])
    print('output_data1.shape:', output_data1.shape)
    print('output_data2.shape:', output_data2.shape)

    print('output_data1:', output_data1)
    print('output_data2:', output_data2)
