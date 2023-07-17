onnx2tf -i rtmpose_wholebody_m_1x3x256x192.onnx -o rtmpose_wholebody_m_1x3x256x192 -osd -cotof -coion -oiqt -qt per-tensor
onnx2tf -i rtmpose_wholebody_l_1x3x256x192.onnx -o rtmpose_wholebody_l_1x3x256x192 -osd  -cotof -coion -oiqt -qt per-tensor
onnx2tf -i rtmpose_wholebody_l_1x3x384x288.onnx -o rtmpose_wholebody_l_1x3x384x288 -osd  -cotof -coion -oiqt -qt per-tensor
onnx2tf -i rtmpose_wholebody_x_1x3x384x288.onnx -o rtmpose_wholebody_x_1x3x384x288 -osd  -cotof -coion -oiqt -qt per-tensor
