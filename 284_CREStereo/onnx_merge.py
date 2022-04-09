import onnx
from onnxsim import simplify

INIT_MODEL = [
    'crestereo_init_iter10_120x160.onnx',
    'crestereo_init_iter10_160x240.onnx',
    'crestereo_init_iter10_180x320.onnx',
    'crestereo_init_iter10_240x320.onnx',
    'crestereo_init_iter10_360x640.onnx',
    'crestereo_init_iter20_120x160.onnx',
    'crestereo_init_iter20_160x240.onnx',
    'crestereo_init_iter20_180x320.onnx',
    'crestereo_init_iter20_240x320.onnx',
    'crestereo_init_iter20_360x640.onnx',
]

NEXT_MODEL = [
    'crestereo_next_iter10_240x320.onnx',
    'crestereo_next_iter10_320x480.onnx',
    'crestereo_next_iter10_360x640.onnx',
    'crestereo_next_iter10_480x640.onnx',
    'crestereo_next_iter10_720x1280.onnx',
    'crestereo_next_iter20_240x320.onnx',
    'crestereo_next_iter20_320x480.onnx',
    'crestereo_next_iter20_360x640.onnx',
    'crestereo_next_iter20_480x640.onnx',
    'crestereo_next_iter20_720x1280.onnx',
]

for init_model, next_model in zip(INIT_MODEL, NEXT_MODEL):
    model1 = onnx.load(init_model)
    model1 = onnx.compose.add_prefix(model1, prefix='init_')

    model2 = onnx.load(next_model)
    model2 = onnx.compose.add_prefix(model2, prefix='next_')
    combined_model = onnx.compose.merge_models(
        model1, model2,
        io_map=[('init_output', 'next_flow_init')]
    )
    file_name = \
        next_model.split('_')[0] + '_' + \
        'combined_' + \
        next_model.split('_')[2] + '_' + \
        next_model.split('_')[3].split('.')[0] + \
        '.onnx'
    onnx.save(combined_model, file_name)
