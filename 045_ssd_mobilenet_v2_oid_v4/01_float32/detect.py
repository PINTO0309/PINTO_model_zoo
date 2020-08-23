import detect_common as common
from PIL import Image
import tflite_runtime.interpreter as tflite
from glob import glob
import pandas as pd

model = 'ssd_mobilenet_v2_oid_v4_300x300_full_integer_quant_edgetpu.tflite'
interpreter = common.make_interpreter(model)
interpreter.allocate_tensors()
label_url = 'https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv'
df = pd.read_csv(label_url)
labels = [row[1] for i, row in df.iterrows()]
img = Image.open('grace_hopper.bmp')
common.set_input(interpreter, img)
interpreter.invoke()
classes = common.output_tensor(interpreter, 1)
scores = common.output_tensor(interpreter, 2)
for i in range(len(classes)):
    print(classes[i], scores[i])