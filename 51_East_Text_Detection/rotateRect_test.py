### https://biotech-lab.org/articles/1251#RotatedRect

import numpy as np
import math
import cv2
  
def rotatedRectangle(img, rotatedRect, color, thickness=1, lineType=cv2.LINE_8, shift=0):
    (x,y), (width, height), angle = rotatedRect
    angle = math.radians(angle)
 
    # 回転する前の矩形の頂点
    pt1_1 = (int(x + width / 2), int(y + height / 2))
    pt2_1 = (int(x + width / 2), int(y - height / 2))
    pt3_1 = (int(x - width / 2), int(y - height / 2))
    pt4_1 = (int(x - width / 2), int(y + height / 2))
 
    # 変換行列
    t = np.array([[np.cos(angle),   -np.sin(angle), x-x*np.cos(angle)+y*np.sin(angle)],
                    [np.sin(angle), np.cos(angle),  y-x*np.sin(angle)-y*np.cos(angle)],
                    [0,             0,              1]])
 
    tmp_pt1_1 = np.array([[pt1_1[0]], [pt1_1[1]], [1]])
    tmp_pt1_2 = np.dot(t, tmp_pt1_1)
    pt1_2 = (int(tmp_pt1_2[0][0]), int(tmp_pt1_2[1][0]))
 
    tmp_pt2_1 = np.array([[pt2_1[0]], [pt2_1[1]], [1]])
    tmp_pt2_2 = np.dot(t, tmp_pt2_1)
    pt2_2 = (int(tmp_pt2_2[0][0]), int(tmp_pt2_2[1][0]))
 
    tmp_pt3_1 = np.array([[pt3_1[0]], [pt3_1[1]], [1]])
    tmp_pt3_2 = np.dot(t, tmp_pt3_1)
    pt3_2 = (int(tmp_pt3_2[0][0]), int(tmp_pt3_2[1][0]))
 
    tmp_pt4_1 = np.array([[pt4_1[0]], [pt4_1[1]], [1]])
    tmp_pt4_2 = np.dot(t, tmp_pt4_1)
    pt4_2 = (int(tmp_pt4_2[0][0]), int(tmp_pt4_2[1][0]))
 
    points = np.array([pt1_2, pt2_2, pt3_2, pt4_2])
    cv2.polylines(img, [points], True, color, thickness, lineType, shift)
    return img
 
img = np.zeros((300, 300, 3), np.uint8)
rotatedRect = ((150, 150), (200, 120), 45)
rotatedRectangle(img, rotatedRect, (255, 255, 0))
 
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()