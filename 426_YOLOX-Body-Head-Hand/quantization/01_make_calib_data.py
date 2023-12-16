import cv2
import glob
import numpy as np

H=128
W=160
# H=256
# W=320
# H=480
# W=640

files = glob.glob("data/*.jpg")
img_datas = []
for idx, file in enumerate(files):
    bgr_img = cv2.imread(file)
    resized_img = cv2.resize(bgr_img, (W, H))
    extend_batch_size_img = resized_img[np.newaxis, :].astype(np.float32)
    print(
        f'{str(idx+1).zfill(2)}. extend_batch_size_img.shape: {extend_batch_size_img.shape}'
    )
    img_datas.append(extend_batch_size_img)
calib_datas = np.vstack(img_datas)
print(f'calib_datas.shape: {calib_datas.shape}')
np.save(file=f'calibdata_bgr_no_norm_{H}x{W}.npy', arr=calib_datas)