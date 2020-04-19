
### tf-nightly-2.2.0.dev20200418

import tensorflow as tf
import json

with tf.compat.v1.Session() as sess:
    with tf.compat.v1.gfile.GFile('./yolov3_nano_256.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        _ = tf.import_graph_def(graph_def)
        ops = {}
        for op in tf.compat.v1.get_default_graph().get_operations():
            ops[op.name] = [str(output) for output in op.outputs]
        with open('./yolov3_nano_256.json', 'w') as f:
            f.write(json.dumps(ops))