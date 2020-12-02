### tensorflow==2.3.1

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_416x416', source='tensorflow')
mlmodel.save("yolov5s_416x416_float32.mlmodel")
