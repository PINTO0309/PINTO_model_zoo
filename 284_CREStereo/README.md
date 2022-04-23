# Note

- Verified results courtesy of ibaiGorordo

  https://github.com/ibaiGorordo/ONNX-CREStereo-Depth-Estimation

  https://github.com/ibaiGorordo/CREStereo-Pytorch

  ![ezgif com-gif-maker (7)](https://user-images.githubusercontent.com/33194443/162555069-449570d2-7476-4d10-ac3b-c50876a63782.gif)

  ![image](https://user-images.githubusercontent.com/33194443/162574481-7d4e9098-0c84-4b7f-9b45-62e312a2c7b6.png)

- Verified results courtesy of yamifuwazaia

  ![image](https://user-images.githubusercontent.com/33194443/162623239-88f1c562-eca6-47d1-aa04-e67d5fbbbafa.png)

- [WIP] OAK-D (Myriad)
  ```bash
  ###
  ### crestereo_init_iter2_120x160.onnx - TensorRT ver.
  ###

  python3 onnx_convert_to_oak-d_myriad.py

  ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py \
  --input_model crestereo_init_iter2_120x160_myriad_oak.onnx \
  --data_type FP16 \
  --output_dir crestereo_init_iter2_120x160_myriad_oak/openvino/FP16 \
  --model_name crestereo_init_iter2_120x160_myriad_oak

  mkdir -p crestereo_init_iter2_120x160_myriad_oak/openvino/myriad

  ${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/myriad_compile \
  -m crestereo_init_iter2_120x160_myriad_oak/openvino/FP16/crestereo_init_iter2_120x160_myriad_oak.xml \
  -ip U8 \
  -VPU_NUMBER_OF_SHAVES 4 \
  -VPU_NUMBER_OF_CMX_SLICES 4 \
  -o crestereo_init_iter2_120x160_myriad_oak/openvino/myriad/crestereo_init_iter2_120x160_myriad_oak.blob
  ```
  ![image](https://user-images.githubusercontent.com/33194443/164913113-5053fb8a-0b48-4a11-85bf-b19123cb6f76.png)
