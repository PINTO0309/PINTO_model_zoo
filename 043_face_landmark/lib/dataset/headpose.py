#-*-coding:utf-8-*-


import cv2
import numpy as np




# object_pts = np.float32([[6.825897, 6.760612, 4.402142],
#                          [1.330353, 7.122144, 6.903745],
#                          [-1.330353, 7.122144, 6.903745],
#                          [-6.825897, 6.760612, 4.402142],
#                          [5.311432, 5.485328, 3.987654],
#                          [1.789930, 5.393625, 4.413414],
#                          [-1.789930, 5.393625, 4.413414],
#                          [-5.311432, 5.485328, 3.987654],
#                          [2.005628, 1.409845, 6.165652],
#                          [-2.005628, 1.409845, 6.165652],
#                          [2.774015, -2.080775, 5.048531],
#                          [-2.774015, -2.080775, 5.048531],
#                          [0.000000, -3.116408, 6.097667],
#                          [0.000000, -7.415691, 4.070434]])
object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652]])
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape,img):
    h,w,_=img.shape
    K = [w, 0.0, w//2,
         0.0, w, h//2,
         0.0, 0.0, 1.0]
    # Assuming no lens distortion
    D = [0, 0, 0.0, 0.0, 0]

    cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
    dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)



    # image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
    #                         shape[39], shape[42], shape[45], shape[31], shape[35],
    #                         shape[48], shape[54], shape[57], shape[8]])
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35]])
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle
