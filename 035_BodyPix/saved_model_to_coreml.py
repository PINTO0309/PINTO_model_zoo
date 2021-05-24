import coremltools as ct

def model_convert(model_name, stride_num, H, W):
    saved_model_path = f'{model_name}/{stride_num}/saved_model_{H}x{W}'
    input = ct.TensorType(name='sub_2', shape=(1, H, W, 3))
    mlmodel = ct.convert(saved_model_path, inputs=[input], source='tensorflow')
    mlmodel.save(f'{saved_model_path}/model_coreml_float32.mlmodel')


model_name = 'mobilenet050'
stride_num = 'stride8'
H = 240
W = 320
model_convert(model_name, stride_num, H, W)
H = 480
W = 640
model_convert(model_name, stride_num, H, W)

model_name = 'mobilenet050'
stride_num = 'stride16'
H = 240
W = 320
model_convert(model_name, stride_num, H, W)
H = 480
W = 640
model_convert(model_name, stride_num, H, W)

model_name = 'mobilenet050'
stride_num = 'stride8'
H = 240
W = 320
model_convert(model_name, stride_num, H, W)
H = 480
W = 640
model_convert(model_name, stride_num, H, W)

#=====================================================

model_name = 'mobilenet075'
stride_num = 'stride8'
H = 240
W = 320
model_convert(model_name, stride_num, H, W)
H = 480
W = 640
model_convert(model_name, stride_num, H, W)

model_name = 'mobilenet075'
stride_num = 'stride16'
H = 240
W = 320
model_convert(model_name, stride_num, H, W)
H = 480
W = 640
model_convert(model_name, stride_num, H, W)

#=====================================================

model_name = 'mobilenet100'
stride_num = 'stride8'
H = 240
W = 320
model_convert(model_name, stride_num, H, W)
H = 480
W = 640
model_convert(model_name, stride_num, H, W)

model_name = 'mobilenet100'
stride_num = 'stride16'
H = 240
W = 320
model_convert(model_name, stride_num, H, W)
H = 480
W = 640
model_convert(model_name, stride_num, H, W)

#=====================================================

model_name = 'resnet50'
stride_num = 'stride16'
H = 240
W = 320
model_convert(model_name, stride_num, H, W)
H = 480
W = 640
model_convert(model_name, stride_num, H, W)

model_name = 'resnet50'
stride_num = 'stride32'
H = 240
W = 320
model_convert(model_name, stride_num, H, W)
H = 480
W = 640
model_convert(model_name, stride_num, H, W)