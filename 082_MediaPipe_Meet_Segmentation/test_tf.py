import numpy as np
from PIL import Image
from utils import load_graph_model, get_input_tensors, get_output_tensors
import tensorflow as tf
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

def load_frozen_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")
    return graph

kWidth = 256
kHeight = 144
tf_model_path = "model_v2.pb"
tflite_model_path = "saved_model_144x256/model_float32.tflite"

print("Loading model...", end="")
graph = load_frozen_graph(tf_model_path)
print("done.\n", end="")

# Get input and output tensors
input_tensor_names = get_input_tensors(graph)
print(input_tensor_names)
output_tensor_names = get_output_tensors(graph)
print(output_tensor_names)

image = Image.open('test.jpg')
h = image.size[1]
w = image.size[0]
img = image.resize((kWidth, kHeight))
img = np.asarray(img)
img = img / 255.
img = img.astype(np.float32)
img = img[np.newaxis,:,:,:]

# Tensorflow Lite
interpreter = Interpreter(tflite_model_path, num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]['index']
output_details = interpreter.get_output_details()[0]['index']

interpreter.set_tensor(input_details, img)
interpreter.invoke()
output = interpreter.get_tensor(output_details)

print(output.shape)
out1 = output[0][:, :, 0]
out2 = output[0][:, :, 1]

out1 = (out1 > 0.5) * 255
out2 = (out2 > 0.5) * 255

print('out1:', out1.shape)
print('out2:', out2.shape)

out1 = Image.fromarray(np.uint8(out1)).resize((w, h))
out2 = Image.fromarray(np.uint8(out2)).resize((w, h))

out1.save('out1.jpg')
out2.save('out2.jpg')

# evaluate the loaded model directly
print("Running inference...", end="")
input_tensor = graph.get_tensor_by_name(input_tensor_names[0])
with tf.compat.v1.Session(graph=graph) as sess:
    results = sess.run(output_tensor_names, feed_dict={
                       input_tensor: img})
print("done. {} outputs received".format(len(results)))

segments = np.squeeze(results[0], 0)
print('segments', segments.shape)
segments1 = segments[:, :, 0]
segments2 = segments[:, :, 1]
segments1 = (segments1 > 0.5) * 255
segments2 = (segments2 > 0.5) * 255

segments1 = Image.fromarray(np.uint8(segments1)).resize((w, h))
segments2 = Image.fromarray(np.uint8(segments2)).resize((w, h))
segments1.save('segments1.jpg')
segments2.save('segments2.jpg')

mask_img = segments2.resize((w, h), Image.LANCZOS).convert("RGB")
mask_img = tf.keras.preprocessing.image.img_to_array(mask_img, dtype=np.uint8)
fg = np.bitwise_and(np.array(image), np.array(mask_img))
fg_img = Image.fromarray(np.uint8(fg)).resize((w, h))
fg_img.save('fg.jpg')
