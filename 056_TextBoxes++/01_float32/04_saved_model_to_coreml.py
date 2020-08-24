### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_256x256', source='tensorflow')
mlmodel.save("dstbpp512fl_sythtext_256x256_float32.mlmodel")

mlmodel = ct.convert('saved_model_320x320', source='tensorflow')
mlmodel.save("dstbpp512fl_sythtext_320x320_float32.mlmodel")

mlmodel = ct.convert('saved_model_416x416', source='tensorflow')
mlmodel.save("dstbpp512fl_sythtext_416x416_float32.mlmodel")

mlmodel = ct.convert('saved_model_512x512', source='tensorflow')
mlmodel.save("dstbpp512fl_sythtext_512x512_float32.mlmodel")