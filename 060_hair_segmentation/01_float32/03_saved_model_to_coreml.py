### tensorflow==2.3.1

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_512x512', source='tensorflow')
mlmodel.save("hair_segmentation_512x512_float32.mlmodel")
