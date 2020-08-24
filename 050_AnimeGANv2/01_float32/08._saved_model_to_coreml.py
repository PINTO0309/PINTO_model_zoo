### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_Hayao', source='tensorflow')
mlmodel.save("animeganv2_hayao_256x256_float32.mlmodel")

mlmodel = ct.convert('saved_model_Paprika', source='tensorflow')
mlmodel.save("animeganv2_paprika_256x256_float32.mlmodel")
