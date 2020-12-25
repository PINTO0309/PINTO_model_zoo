### tensorflow==2.3.1

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_from_pb', source='tensorflow')
mlmodel.save("yolact_550x550_opt_float32.mlmodel")
