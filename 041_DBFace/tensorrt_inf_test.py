### tensorflow==2.3.0

### https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/6.0/GA_6.0.1.5/local_repos/nv-tensorrt-repo-ubuntu1804-cuda10.1-trt6.0.1.5-ga-20190913_1-1_amd64.deb

# os="ubuntu1804"
# tag="cuda10.1-trt6.0.1.5-ga-20190913"
# sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
# sudo apt-key add /var/nv-tensorrt-repo-${tag}/7fa2af80.pub

# sudo apt-get update
# sudo apt-get install tensorrt

# sudo apt-get install python3-libnvinfer-dev
# python3-libnvinfer
# sudo apt-get install uff-converter-tf
# sudo apt-get install onnx-graphsurgeon

import tensorflow as tf
import numpy as np
import time

# from tensorflow.python.compiler.tensorrt import trt_convert as trt

# converter = trt.TrtGraphConverterV2(input_saved_model_dir='saved_model_keras_480x640',
#                                     conversion_params=trt.DEFAULT_TRT_CONVERSION_PARAMS)
# converter.convert()
# converter.save('dbface_keras_480x640_tensorrt_float32_nhwc')

# converter = trt.TrtGraphConverterV2(input_saved_model_dir='saved_model_keras_480x640',
#                                     conversion_params=trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP16))
# converter.convert()
# converter.save('dbface_keras_480x640_tensorrt_float16_nhwc')


def input_fn():
    input_shapes = [1, 480, 640, 3]
    yield [np.zeros(input_shapes).astype(np.float32)]

params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP32', maximum_cached_engines=1000)
converter = tf.experimental.tensorrt.Converter(input_saved_model_dir='saved_model_keras_480x640', conversion_params=params)
converter.convert()
converter.build(input_fn=input_fn)
converter.save('dbface_keras_480x640_tensorrt_float32_nhwc')


params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP16', maximum_cached_engines=1000)
converter = tf.experimental.tensorrt.Converter(input_saved_model_dir='saved_model_keras_480x640', conversion_params=params)
converter.convert()
converter.build(input_fn=input_fn)
converter.save('dbface_keras_480x640_tensorrt_float16_nhwc')



model = tf.saved_model.load('dbface_keras_480x640_tensorrt_float32_nhwc', tags=[tf.saved_model.SERVING])
infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
infer.inputs[0].shape
x = np.random.uniform(size=(1,480,640,3)).astype(np.float32)

start = time.perf_counter()
infer(tf.convert_to_tensor(x))
end = time.perf_counter()
print('@@@@@@@@@@@@@@ First Inference')
print('elapsed time:', end - start)

start = time.perf_counter()
infer(tf.convert_to_tensor(x))
end = time.perf_counter()
print('@@@@@@@@@@@@@@ Second Inference')
print('elapsed time:', end - start)


model = tf.saved_model.load('dbface_keras_480x640_tensorrt_float16_nhwc', tags=[tf.saved_model.SERVING])
infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
infer.inputs[0].shape
x = np.random.uniform(size=(1,480,640,3)).astype(np.float32)

start = time.perf_counter()
infer(tf.convert_to_tensor(x))
end = time.perf_counter()
print('@@@@@@@@@@@@@@ First Inference')
print('elapsed time:', end - start)

start = time.perf_counter()
infer(tf.convert_to_tensor(x))
end = time.perf_counter()
print('@@@@@@@@@@@@@@ Second Inference')
print('elapsed time:', end - start)
