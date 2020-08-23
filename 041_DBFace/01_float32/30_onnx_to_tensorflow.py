import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load('dbface.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('dbface_tf.pb')