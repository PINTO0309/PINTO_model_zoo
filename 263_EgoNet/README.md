## Results of logic analysis when converting EgoNet models
![1](https://user-images.githubusercontent.com/33194443/156893046-0f514c27-5899-4637-b284-e660faeb235b.png) ![2](https://user-images.githubusercontent.com/33194443/156893059-ef96afc7-df8e-4322-aa61-98e0e432e916.png)
1. This model requires two different files to be run in sequence. The order of execution is **`egonet_heatmap_Nx3x256x256`** first, then **`egonet_fc_Nx66`**.
2. **`N`** in the input tensor of **`egonet_heatmap_Nx3x256x256`** is the batch size. What this batch size means is the number of cars detected using your favorite object detection model. This means that before using EgoNet's model, the car must be detected beforehand using an object detection model such as YOLO or MobileNet, and the car region must be cut out with a bounding box. Thus, the batch size **`N`** of the input tensor of the EgoNet model is the number of vehicles obtained by object detection.
3. Preprocessing
    - crop_instances
      1. aaa
      2. bbb
      3. ccc 
4. The normalization process is added at the beginning of the **`egonet_fc_Nx66`** model and the inverse normalization process is added at the end of **`egonet_fc_Nx66`** model. Therefore, the tensor **`[N, 66]`** used as input for the **`egonet_fc_Nx66`** model should not be programmatically normalized.

   To be more specific, the following two lines should not be executed programmatically. The quoted process reads the .npy and uses it for normalization, but the model committed to my repository has already fused the .npy values into the model.
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
    ```
    ![image](https://user-images.githubusercontent.com/33194443/156892669-6db296f6-fe11-44ad-abde-723893ad38f1.png)
5. Intermediate processing is required after **`egonet_heatmap_Nx3x256x256`** before passing **`local_coord`** values to **`egonet_fc_Nx66`**. The process of affine transforming the value of the output of **`egonet_heatmap_Nx3x256x256`**. The affine transformed values are saved as **`kpts_2d_pred`** and passed on to the next model input.
https://github.com/Nicholasli1995/EgoNet/blob/a3ea8285d0497723dc2a3a60009b2da95937f542/libs/model/egonet.py#L424-L467
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
        output = self.HC(instances)
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
