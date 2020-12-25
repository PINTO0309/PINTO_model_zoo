### tensorflow==2.3.1

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_256x256', source='tensorflow')
mlmodel.save("nanodet_256x256_float32.mlmodel")

mlmodel = ct.convert('saved_model_320x320', source='tensorflow')
mlmodel.save("nanodet_320x320_float32.mlmodel")

mlmodel = ct.convert('saved_model_416x416', source='tensorflow')
mlmodel.save("nanodet_416x416_float32.mlmodel")

mlmodel = ct.convert('saved_model_512x512', source='tensorflow')
mlmodel.save("nanodet_512x512_float32.mlmodel")

mlmodel = ct.convert('saved_model_640x640', source='tensorflow')
mlmodel.save("nanodet_640x640_float32.mlmodel")