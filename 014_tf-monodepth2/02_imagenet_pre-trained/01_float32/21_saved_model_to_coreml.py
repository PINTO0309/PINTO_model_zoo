### tensorflow==2.3.1

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_colormap_depth_nopt', source='tensorflow')
mlmodel.save("colormap_depth_nopt_192x640_float32.mlmodel")

mlmodel = ct.convert('saved_model_colormap_depth_pt', source='tensorflow')
mlmodel.save("colormap_depth_pt_192x640_float32.mlmodel")


mlmodel = ct.convert('saved_model_colormap_only_nopt', source='tensorflow')
mlmodel.save("colormap_only_nopt_192x640_float32.mlmodel")

mlmodel = ct.convert('saved_model_colormap_only_pt', source='tensorflow')
mlmodel.save("colormap_only_pt_192x640_float32.mlmodel")