### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model', source='tensorflow')
mlmodel.save("yolov4_tiny_voc_416x416_float32.mlmodel")

