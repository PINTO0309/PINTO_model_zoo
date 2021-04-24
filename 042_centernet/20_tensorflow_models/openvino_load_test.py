from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
import pprint

ie = IECore()
vino_model_path = f'saved_model/openvino/FP16'
vino_model = 'saved_model'
net = ie.read_network(model=f'{vino_model_path}/{vino_model}.xml', weights=f'{vino_model_path}/{vino_model}.bin')
exec_net = ie.load_network(network=net, device_name='CPU', num_requests=2)

input_blob = next(iter(net.input_info))
out_blob = [o for o in net.outputs]

pprint.pprint('input_blob:')
pprint.pprint(input_blob)
pprint.pprint('out_blob:')
pprint.pprint(out_blob)

cap = cv2.VideoCapture('person.png')
ret, frame = cap.read()
frame_h, frame_w = frame.shape[:2]

width = 320
height = 320
im = cv2.resize(frame.copy(), (width, height))
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = im.transpose((2, 0, 1))
im = im[np.newaxis, :, :, :]

inputs = {input_blob: im}
exec_net.requests[0].wait(-1)
exec_net.start_async(request_id=0, inputs=inputs)
if exec_net.requests[0].wait(-1) == 0:
    res = [
        exec_net.requests[0].output_blobs[out_blob[0]].buffer,
        exec_net.requests[0].output_blobs[out_blob[1]].buffer,
        exec_net.requests[0].output_blobs[out_blob[2]].buffer,
        exec_net.requests[0].output_blobs[out_blob[3]].buffer,
        exec_net.requests[0].output_blobs[out_blob[4]].buffer,
        exec_net.requests[0].output_blobs[out_blob[5]].buffer,
    ]
    pprint.pprint('res:')
    pprint.pprint(out_blob[0])
    pprint.pprint(res[0].shape)
    pprint.pprint(res[0])
    pprint.pprint(out_blob[1])
    pprint.pprint(res[1].shape)
    pprint.pprint(res[1])
    pprint.pprint(out_blob[2])
    pprint.pprint(res[2].shape)
    pprint.pprint(res[2])
    pprint.pprint(out_blob[3])
    pprint.pprint(res[3].shape)
    pprint.pprint(res[3])
    pprint.pprint(out_blob[4])
    pprint.pprint(res[4].shape)
    pprint.pprint(res[4])
    pprint.pprint(out_blob[5])
    pprint.pprint(res[5].shape)
    pprint.pprint(res[5])

    person = res[5][0][0]
    print('person:', person)
    ymin = int(person[0] * frame_h)
    xmin = int(person[1] * frame_w)
    ymax = int(person[2] * frame_h)
    xmax = int(person[3] * frame_w)
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0))

    KEYPOINT_EDGES = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    print('res1:', res[1].shape)
    bone_y = res[1][0][0][0]
    bone_x = res[1][0][1][0]
    print('bone_x.shape:', bone_x.shape)
    print('bone_x:', bone_x)
    print('bone_y.shape:', bone_y.shape)
    print('bone_y:', bone_y)

    for keypoint_x, keypoint_y in zip(bone_x, bone_y):
        cv2.circle(
            frame,
            (int(keypoint_x * frame_w), int(keypoint_y * frame_h)),
            2,
            (0, 255, 0))

    for keypoint_start, keypoint_end in KEYPOINT_EDGES:
        cv2.line(
            frame,
            (int(bone_x[keypoint_start] * frame_w), int(bone_y[keypoint_start] * frame_h)),
            (int(bone_x[keypoint_end] * frame_w),   int(bone_y[keypoint_end] * frame_h)),
            (0, 255, 0),
            2)

    cv2.namedWindow('centernet')
    cv2.imshow('centernet', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()