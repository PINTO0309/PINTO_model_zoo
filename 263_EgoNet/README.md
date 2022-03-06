## Results of logic analysis when converting EgoNet models
![1](https://user-images.githubusercontent.com/33194443/156893046-0f514c27-5899-4637-b284-e660faeb235b.png) ![2](https://user-images.githubusercontent.com/33194443/156893059-ef96afc7-df8e-4322-aa61-98e0e432e916.png)
1. This model requires two different files to be run in sequence. The order of execution is **`egonet_heatmap_Nx3x256x256`** first, then **`egonet_fc_Nx66`**.
2. **`N`** in the input tensor of **`egonet_heatmap_Nx3x256x256`** is the batch size. What this batch size means is the number of cars detected using your favorite object detection model. This means that before using EgoNet's model, the car must be detected beforehand using an object detection model such as YOLO or SSD, etc, and the car region must be cut out with a bounding box. Thus, the batch size **`N`** of the input tensor of the EgoNet model is the number of vehicles obtained by object detection.
3. Preprocessing
    - crop_instances - Loop processing for all detected car bounding boxes
      1. INPUT bbox:  
        `[[654 , 184, 701, 232]] <- [[X1, Y1, X2, Y2],[X1, Y1, X2, Y2], ..., [X1, Y1, X2, Y2]]`
      3. INPUT labels:  
        `[-1] <- [-1, -1, ..., -1]` (Cars only, Car = -1)
      4. INPUT scores:  
        `[0.99902] <- [conf1, conf2, ..., confN]`
      5. Resize bbox (Extended car cutout area to 1.1 times larger):  
        `new_width = (701 - 654) * 1.1`  
        `new_height = (232 - 184) * 1.1`  
      6. Calculation of the center point:  
        `center_x = (654 + 701) / 2`  
        `center_y = (184 + 232) / 2`  
      7. Calculation of the new X1, Y1, X2, Y2 point:  
        `new_left = center_x - (0.5 * new_width)`  
        `new_right = center_x + (0.5 * new_width)`  
        `new_top = center_y - (0.5 * new_height)`  
        `new_bottom = center_y + (0.5 * new_height)`  
      8. Resize according to aspect ratio  
          `target_ar` is the aspect ratio calculated by `height / width`.  
          https://github.com/Nicholasli1995/EgoNet/blob/a3ea8285d0497723dc2a3a60009b2da95937f542/libs/common/img_proc.py#L453-L459
          
          ```python
          ret = resize_bbox(new_left, new_top, new_right, new_bottom, target_ar=target_ar)
          
          def resize_bbox(left, top, right, bottom, target_ar=1.):
              """
              Resize a bounding box to pre-defined aspect ratio.
              """
              width = right - left
              height = bottom - top
              aspect_ratio = height/width
              center_x = (left + right)/2
              center_y = (top + bottom)/2
              if aspect_ratio > target_ar:
                  new_width = height*(1/target_ar)
                  new_left = center_x - 0.5*new_width
                  new_right = center_x + 0.5*new_width
                  new_top = top
                  new_bottom = bottom
              else:
                  new_height = width*target_ar
                  new_left = left
                  new_right = right
                  new_top = center_y - 0.5*new_height
                  new_bottom = center_y + 0.5*new_height
              return {
                  'bbox': [new_left, new_top, new_right, new_bottom],
                  'c': np.array([center_x, center_y]),
                  's': np.array([(new_right - new_left)/SIZE, (new_bottom - new_top)/SIZE])
              }
          ```
      9. Affine transformation of vertices  
          `img` is not the detected image of the car, but the whole image that is processed for object detection.
          https://github.com/Nicholasli1995/EgoNet/blob/a3ea8285d0497723dc2a3a60009b2da95937f542/libs/model/egonet.py#L81-L89
          ```python
          c, s, r = ret['c'], ret['s'], 0.0
          trans = get_affine_transform(
              c,
              s,
              r,
              (height, width)
          )
          instance = cv2.warpAffine(
              img,
              trans,
              (int(resolution[0]), int(resolution[1])),
              flags=cv2.INTER_LINEAR
          )
          instance = instance if pth_trans is None else pth_trans(instance)
          ```
          ```
          pth_trans: Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          ```
          https://github.com/Nicholasli1995/EgoNet/blob/a3ea8285d0497723dc2a3a60009b2da95937f542/libs/common/img_proc.py#L26-L64  
          `rot` is rotaion.
          ```python
          def get_affine_transform(
              center,
              scale,
              rot,
              output_size,
              shift=np.array([0, 0], dtype=np.float32),
              inv=0
          ):
              """
              Estimate an affine transformation given crop
              parameters (center, scale and rotation) and output resolution.
              """
              if isinstance(scale, list):
                  scale = np.array(scale)
              if isinstance(center, list):
                  center = np.array(center)
              scale_tmp = scale * SIZE # SIZE=200.0
              src_w = scale_tmp[0]
              dst_h, dst_w = output_size

              rot_rad = np.pi * rot / 180
              src_dir = get_dir([0, src_w * -0.5], rot_rad)
              dst_dir = np.array([0, dst_w * -0.5], np.float32)

              src = np.zeros((3, 2), dtype=np.float32)
              dst = np.zeros((3, 2), dtype=np.float32)
              src[0, :] = center + scale_tmp * shift
              src[1, :] = center + src_dir + scale_tmp * shift
              dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
              dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

              src[2:, :] = get_3rd_point(src[0, :], src[1, :])
              dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

              if inv:
                  trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
              else:
                  trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

              return trans
          ```
          ```python
          def get_dir(src_point, rot_rad):
              sn, cs = np.sin(rot_rad), np.cos(rot_rad)

              src_result = [0, 0]
              src_result[0] = src_point[0] * cs - src_point[1] * sn
              src_result[1] = src_point[0] * sn + src_point[1] * cs

              return src_result
          ```
          ```python
          def get_3rd_point(a, b):
              direct = a - b
              return b + np.array([-direct[1], direct[0]], dtype=np.float32)
          ```
      10. Intermediate processing is required after **`egonet_heatmap_Nx3x256x256`** before passing **`local_coord`** values to **`egonet_fc_Nx66`**. The process of affine transforming the value of the output of **`egonet_heatmap_Nx3x256x256`**. The affine transformed values are saved as **`kpts_2d_pred`** and passed on to the next model input. `instances` is the coordinate information of N cars after affine transformations and normalization.
          https://github.com/Nicholasli1995/EgoNet/blob/a3ea8285d0497723dc2a3a60009b2da95937f542/libs/model/egonet.py#L424-L467
          ```python
          instances =
          tensor([[[-1.5699, -1.5870, -1.5870,  ..., -0.3541, -0.3541, -0.3541],
                  [-1.5699, -1.5699, -1.5870,  ..., -0.3369, -0.3369, -0.3541],
                  [-1.5699, -1.5699, -1.5699,  ..., -0.3369, -0.3369, -0.3369],
                  ...,
                  [ 2.2489,  2.2489,  2.2489,  ...,  1.3413,  1.3242,  1.3242],
                  [ 2.2489,  2.2489,  2.2489,  ...,  1.3927,  1.3755,  1.3927],
                  [ 2.2489,  2.2489,  2.2489,  ...,  1.4269,  1.4269,  1.4440]],

                  [[-1.4930, -1.5105, -1.5105,  ..., -0.1625, -0.1625, -0.1800],
                  [-1.4755, -1.4930, -1.4930,  ..., -0.1275, -0.1275, -0.1275],
                  [-1.4580, -1.4755, -1.4930,  ..., -0.0924, -0.0924, -0.0924],
                  ...,
                  [ 2.3761,  2.3761,  2.3936,  ...,  1.0980,  1.0280,  1.0280],
                  [ 2.3235,  2.3410,  2.3585,  ...,  1.1331,  1.0805,  1.0805],
                  [ 2.2710,  2.3060,  2.3235,  ...,  1.1856,  1.1155,  1.1331]],

                  [[-1.1596, -1.1596, -1.1770,  ...,  0.1476,  0.1476,  0.1476],
                  [-1.1421, -1.1596, -1.1596,  ...,  0.2348,  0.2348,  0.2522],
                  [-1.1247, -1.1421, -1.1596,  ...,  0.3219,  0.3219,  0.3219],
                  ...,
                  [ 2.3437,  2.3263,  2.3088,  ...,  1.2282,  1.1585,  1.1585],
                  [ 2.2914,  2.2740,  2.2740,  ...,  1.2457,  1.1585,  1.1411],
                  [ 2.2391,  2.2391,  2.2217,  ...,  1.2631,  1.1759,  1.1411]]])
          ```
          ```python
          records =
          [
              {
                  'bbox': array(
                      [
                          654.4174967,
                          184.97550979,
                          701.93145589,
                          232.48946898
                      ]
                  ),
                  'bbox_resize': [
                      652.0417987381174,
                      182.59981183069667,
                      704.3071538461882,
                      234.86516693876743
                  ],
                  'center': array(
                      [
                          678.17447629,
                          208.73248938
                      ]
                  ),
                  'label': -1,
                  'path': '/xxx/kitti/training/image_2/000002.png',
                  'rotation': 0.0,
                  'scale': array(
                      [
                          0.26132678,
                          0.26132678
                      ]
                  ),
                  'score': 0.99902
              }
          ]
          ```
          Inverse affine transformation of the keypoints inferred and output by `output = self.HC(instances)`. The converted (preprocessed) keypoints are stored in **`kpts_2d_pred`**. `self.resolution` is `[256, 256]`. `np.array(self.resolution).reshape(1, 1, 2)` is `[[[256, 256]]]`. `local_coord` is `[N, 33, 2]`.
          ```python
          def get_keypoints(
              self,
              instances,
              records,
              is_cuda=True
          ):
              """
              Foward pass to obtain the screen coordinates.
              """
              if is_cuda:
                  instances = instances.cuda()
              output = self.HC(instances) # <--- Heat map generation (inference), egonet_heatmap_Nx3x256x256
              
              # local part coordinates
              width, height = self.resolution
              local_coord = output[1].data.cpu().numpy()
              local_coord *= np.array(self.resolution).reshape(1, 1, 2)
              # transform local part coordinates to screen coordinates
              centers = [records[i]['center'] for i in range(len(records))]
              scales = [records[i]['scale'] for i in range(len(records))]
              rots = [records[i]['rotation'] for i in range(len(records))]
              for instance_idx in range(len(local_coord)):
                  trans_inv = get_affine_transform(
                      centers[instance_idx],
                      scales[instance_idx],
                      rots[instance_idx],
                      (height, width),
                      inv=1
                  )
                  screen_coord = affine_transform_modified(
                      local_coord[instance_idx],
                      trans_inv
                  )
                  records[instance_idx]['kpts'] = screen_coord
              # assemble a dictionary where each key corresponds to one image
              ret = {}
              for record in records:
                  path = record['path']
                  if path not in ret:
                      ret[path] = self.new_img_dict()
                  ret[path]['kpts_2d_pred'].append(record['kpts'].reshape(1, -1))
                  ret[path]['center'].append(record['center'])
                  ret[path]['scale'].append(record['scale'])
                  ret[path]['bbox_resize'].append(record['bbox_resize'])
                  ret[path]['label'].append(record['label'])
                  ret[path]['score'].append(record['score'])
                  ret[path]['rotation'].append(record['rotation'])
              return ret
          ```
          ```python
          def affine_transform_modified(pts, t):
              """
              Apply affine transformation with homogeneous coordinates.
              """
              # pts of shape [n, 2]
              new_pts = np.hstack([pts, np.ones((len(pts), 1))]).T
              new_pts = t @ new_pts
              return new_pts[:2, :].T
          ```
4. The normalization process is added at the beginning of the **`egonet_fc_Nx66`** model and the inverse normalization process is added at the end of **`egonet_fc_Nx66`** model. Therefore, the tensor **`[N, 66]`** used as input for the **`egonet_fc_Nx66`** model should not be programmatically normalized. `len(prediction)` is the batch size `N`. It is synonymous with the number of cars detected for one image.
    https://github.com/Nicholasli1995/EgoNet/blob/a3ea8285d0497723dc2a3a60009b2da95937f542/libs/model/egonet.py#L469-L486
    ```python
    def lift_2d_to_3d(self, records, cuda=True):
        """
        Foward-pass of the lifter sub-model.
        """      
        for path in records.keys():
            data = np.concatenate(records[path]['kpts_2d_pred'], axis=0)
            data = nop.normalize_1d(data, self.LS['mean_in'], self.LS['std_in'])
            data = data.astype(np.float32)
            data = torch.from_numpy(data)
            if cuda:
                data = data.cuda()
            
            prediction = self.L(data) # <--- egonet_fc_Nx66
            
            prediction = nop.unnormalize_1d(
                prediction.data.cpu().numpy(),
                self.LS['mean_out'],
                self.LS['std_out']
            )
            records[path]['kpts_3d_pred'] = prediction.reshape(len(prediction), -1, 3)
        return records
    ```

   To be more specific, the following two lines should not be executed programmatically. The quoted process reads the **`self.LS (.npy)`** and uses it for normalization, but the model committed to my repository has already fused the .npy values into the model.
https://github.com/Nicholasli1995/EgoNet/blob/a3ea8285d0497723dc2a3a60009b2da95937f542/libs/model/egonet.py#L475
    ```python
    data = nop.normalize_1d(data, self.LS['mean_in'], self.LS['std_in'])
    ```
    ![image](https://user-images.githubusercontent.com/33194443/156892608-a0b17195-e706-4fa1-8f43-ab33508d2f3a.png)
    https://github.com/Nicholasli1995/EgoNet/blob/a3ea8285d0497723dc2a3a60009b2da95937f542/libs/model/egonet.py#L481-L484
    ```python
    prediction = nop.unnormalize_1d(
        prediction.data.cpu().numpy(),
        self.LS['mean_out'],
        self.LS['std_out']
    )
    records[path]['kpts_3d_pred'] = prediction.reshape(len(prediction), 32, 3)
    ```
    ![image](https://user-images.githubusercontent.com/33194443/156892669-6db296f6-fe11-44ad-abde-723893ad38f1.png)
