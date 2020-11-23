### tf_nightly-2.5.0.dev20201123

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_320x320', source='tensorflow')
mlmodel.save("nanodet_320x320_float32.mlmodel")

mlmodel = ct.convert('saved_model_416x416', source='tensorflow')
mlmodel.save("nanodet_416x416_float32.mlmodel")