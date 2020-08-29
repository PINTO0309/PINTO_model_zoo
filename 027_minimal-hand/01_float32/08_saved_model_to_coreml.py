### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_detnet', source='tensorflow')
mlmodel.save("detnet_128x128_float32.mlmodel")

mlmodel = ct.convert('saved_model_iknet', source='tensorflow')
mlmodel.save("iknet_float32.mlmodel")
