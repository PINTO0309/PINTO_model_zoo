import onnx

model1 = onnx.load('hep_lum_eng_HxW.onnx')
model1 = onnx.compose.add_prefix(model1, prefix='lum_')

model2 = onnx.load('hep_ndm_eng_encoder_HxW.onnx')
model2 = onnx.compose.add_prefix(model2, prefix='ndm_enc_')
combined_model1 = onnx.compose.merge_models(
    model1, model2,
    io_map=[('lum_227', 'ndm_enc_input.1')]
)

model3 = onnx.load('hep_ndm_eng_decoder_HxW.onnx')
model3 = onnx.compose.add_prefix(model3, prefix='ndm_dec_')
combined_model2 = onnx.compose.merge_models(
    combined_model1, model3,
    io_map=[('ndm_enc_901', 'ndm_dec_input.1')]
)

onnx.save(combined_model2, 'hep_combine_eng_HxW.onnx')
