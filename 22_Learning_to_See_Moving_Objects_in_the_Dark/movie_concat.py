	
### Reference article: https://watlab-blog.com/2019/09/29/movie-space-combine/

import cv2
 
def image_hcombine(im_info1, im_info2):
    img1 = im_info1[0]
    img2 = im_info2[0]
    color_flag1 = im_info1[1]
    color_flag2 = im_info2[1]
 
    if color_flag1 == 1:
        h1, w1, ch1 = img1.shape[:3]
    else:
        h1, w1 = img1.shape[:2]
 
    if color_flag2 == 1:
        h2, w2, ch2 = img2.shape[:3]
    else:
        h2, w2 = img2.shape[:2]
 
    if h1 < h2:
        h1 = h2
        w1 = int((h2 / h1) * w2)
        img1 = cv2.resize(img1, (w1, h1))
    else:
        h2 = h1
        w2 = int((h1 / h2) * w1)
        img2 = cv2.resize(img2, (w2, h2))
 
    img = cv2.hconcat([img1, img2])
    return img
 
def m_space_hcombine(movie1, movie2, path_out, scale_factor):
    path1 = movie1[0]
    path2 = movie2[0]
    color_flag1 = movie1[1]
    color_flag2 = movie2[1]
 
    movie1_obj = cv2.VideoCapture(path1)
    movie2_obj = cv2.VideoCapture(path2)
 
    i = 0
    while True:
        ret1, frame1 = movie1_obj.read()
        ret2, frame2 = movie2_obj.read()
        check = ret1 and ret2
        if check == True:
            im_info1 = [frame1, color_flag1]
            im_info2 = [frame2, color_flag2]
 
            frame_mix = image_hcombine(im_info1, im_info2)
 
            if i == 0:
                fps = int(movie1_obj.get(cv2.CAP_PROP_FPS))
                fps_new = int(fps * scale_factor)
                frame_size = frame_mix.shape[:3]
                h = frame_size[0]
                w = frame_size[1]
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                video = cv2.VideoWriter(path_out, fourcc, fps_new, (w, h))
                i = i + 1 
            else:
                pass
            video.write(frame_mix)
        else:
            break
 
    movie1_obj.release()
    movie2_obj.release()
    return
 
movie1 = ['gain15_view.mp4', True]
movie2 = ['Cam2_gain15.mp4', True]
path_out = 'movie_out.mp4'
scale_factor = 1

m_space_hcombine(movie1, movie2, path_out, scale_factor)