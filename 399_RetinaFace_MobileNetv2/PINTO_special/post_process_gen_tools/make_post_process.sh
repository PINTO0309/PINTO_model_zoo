#!/bin/bash

# pip install -U pip \
# && pip install onnxsim
# && pip install -U simple-onnx-processing-tools \
# && pip install -U onnx \
# && python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
# && pip install tensorflow==2.12.0

OPSET=11
BATCHES=1
CLASSES=1

MAX_OUTPUT_BOXES_PER_CLASS=1000
IOU_THRESHOLD=0.40
SCORE_THRESHOLD=0.30

RESOLUTIONS=(
    "180 320 2440"
    "180 416 3172"
    "180 512 3904"
    "180 640 4880"
    "180 800 6100"
    "240 320 3160"
    "240 416 4108"
    "240 512 5056"
    "240 640 6320"
    "240 800 7900"
    "240 960 9480"
    "288 1280 15120"
    "288 480 5670"
    "288 512 6048"
    "288 640 7560"
    "288 800 9450"
    "288 960 11340"
    "320 320 4200"
    "360 1280 19040"
    "360 480 7140"
    "360 512 7616"
    "360 640 9520"
    "360 800 11900"
    "360 960 14280"
    "376 1344 20832"
    "416 416 7098"
    "480 1280 25200"
    "480 640 12600"
    "480 800 15750"
    "480 960 18900"
    "512 512 10752"
    "540 1280 28560"
    "540 800 17850"
    "540 960 21420"
    "640 640 16800"
    "640 960 25200"
    "720 1280 37840"
    "720 2560 75680"
    "1080 1920 85200"
)

for((i=0; i<${#RESOLUTIONS[@]}; i++))
do
    RESOLUTION=(`echo ${RESOLUTIONS[i]}`)
    IMAGE_HEIGHT=${RESOLUTION[0]}
    IMAGE_WIDTH=${RESOLUTION[1]}
    BOXES=${RESOLUTION[2]}

    ################################################### Boxes + Scores
    python make_boxes_scores.py -o ${OPSET} -b ${BATCHES} -x ${BOXES} -c ${CLASSES}
    python make_cxcywh_y1x1y2x2.py -o ${OPSET} -b ${BATCHES} -x ${BOXES}

    snc4onnx \
    --input_onnx_file_paths 02_boxes_scores_${BOXES}.onnx 03_cxcywh_y1x1y2x2_${BOXES}.onnx \
    --srcop_destop boxes_cxcywh cxcywh \
    --op_prefixes_after_merging 02 03 \
    --output_onnx_file_path 04_boxes_x1y1x2y2_y1x1y2x2_scores_${BOXES}.onnx


    ################################################### NonMaxSuppression
    sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name max_output_boxes_per_class_const \
    --output_variables max_output_boxes_per_class int64 [1] \
    --attributes value int64 [${MAX_OUTPUT_BOXES_PER_CLASS}] \
    --output_onnx_file_path 05_Constant_max_output_boxes_per_class.onnx

    sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name iou_threshold_const \
    --output_variables iou_threshold float32 [1] \
    --attributes value float32 [${IOU_THRESHOLD}] \
    --output_onnx_file_path 06_Constant_iou_threshold.onnx

    sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name score_threshold_const \
    --output_variables score_threshold float32 [1] \
    --attributes value float32 [${SCORE_THRESHOLD}] \
    --output_onnx_file_path 07_Constant_score_threshold.onnx


    OP=NonMaxSuppression
    LOWEROP=${OP,,}
    sog4onnx \
    --op_type ${OP} \
    --opset ${OPSET} \
    --op_name ${LOWEROP}${OPSET} \
    --input_variables boxes_var float32 [${BATCHES},${BOXES},4] \
    --input_variables scores_var float32 [${BATCHES},${CLASSES},${BOXES}] \
    --input_variables max_output_boxes_per_class_var int64 [1] \
    --input_variables iou_threshold_var float32 [1] \
    --input_variables score_threshold_var float32 [1] \
    --output_variables selected_indices int64 [\'N\',3] \
    --attributes center_point_box int64 0 \
    --output_onnx_file_path 08_${OP}${OPSET}.onnx


    snc4onnx \
    --input_onnx_file_paths 05_Constant_max_output_boxes_per_class.onnx 08_${OP}${OPSET}.onnx \
    --srcop_destop max_output_boxes_per_class max_output_boxes_per_class_var \
    --output_onnx_file_path 08_${OP}${OPSET}.onnx

    snc4onnx \
    --input_onnx_file_paths 06_Constant_iou_threshold.onnx 08_${OP}${OPSET}.onnx \
    --srcop_destop iou_threshold iou_threshold_var \
    --output_onnx_file_path 08_${OP}${OPSET}.onnx

    snc4onnx \
    --input_onnx_file_paths 07_Constant_score_threshold.onnx 08_${OP}${OPSET}.onnx \
    --srcop_destop score_threshold score_threshold_var \
    --output_onnx_file_path 08_${OP}${OPSET}.onnx

    soc4onnx \
    --input_onnx_file_path 08_${OP}${OPSET}.onnx \
    --output_onnx_file_path 08_${OP}${OPSET}.onnx \
    --opset ${OPSET}


    ################################################### Boxes + Scores + NonMaxSuppression
    snc4onnx \
    --input_onnx_file_paths 04_boxes_x1y1x2y2_y1x1y2x2_scores_${BOXES}.onnx 08_${OP}${OPSET}.onnx \
    --srcop_destop scores scores_var y1x1y2x2 boxes_var \
    --output_onnx_file_path 09_nms_retinaface_${BOXES}.onnx


    ################################################### Myriad workaround Mul
    OP=Mul
    LOWEROP=${OP,,}
    OPSET=${OPSET}
    sog4onnx \
    --op_type ${OP} \
    --opset ${OPSET} \
    --op_name ${LOWEROP}${OPSET} \
    --input_variables workaround_mul_a int64 [\'N\',3] \
    --input_variables workaround_mul_b int64 [1] \
    --output_variables workaround_mul_out int64 [\'N\',3] \
    --output_onnx_file_path 10_${OP}${OPSET}_workaround.onnx


    ############ Myriad workaround Constant
    sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name workaround_mul_const_op \
    --output_variables workaround_mul_const int64 [1] \
    --attributes value int64 [1] \
    --output_onnx_file_path 11_Constant_workaround_mul.onnx

    ############ Myriad workaround Mul + Myriad workaround Constant
    snc4onnx \
    --input_onnx_file_paths 11_Constant_workaround_mul.onnx 10_${OP}${OPSET}_workaround.onnx \
    --srcop_destop workaround_mul_const workaround_mul_b \
    --output_onnx_file_path 11_Constant_workaround_mul.onnx



    ################################################### NonMaxSuppression + Myriad workaround Mul
    snc4onnx \
    --input_onnx_file_paths 09_nms_retinaface_${BOXES}.onnx 11_Constant_workaround_mul.onnx \
    --srcop_destop selected_indices workaround_mul_a \
    --disable_onnxsim \
    --output_onnx_file_path 09_nms_retinaface_${BOXES}.onnx



    ################################################### Score GatherND
    python make_score_gather_nd.py -b ${BATCHES} -x ${BOXES} -c ${CLASSES}

    python -m tf2onnx.convert \
    --opset ${OPSET} \
    --tflite saved_model_postprocess/nms_score_gather_nd.tflite \
    --output 12_nms_score_gather_nd.onnx

    sor4onnx \
    --input_onnx_file_path 12_nms_score_gather_nd.onnx \
    --old_new ":0" "" \
    --search_mode "suffix_match" \
    --output_onnx_file_path 12_nms_score_gather_nd.onnx

    sor4onnx \
    --input_onnx_file_path 12_nms_score_gather_nd.onnx \
    --old_new "serving_default_input_1" "gn_scores" \
    --output_onnx_file_path 12_nms_score_gather_nd.onnx \
    --mode inputs

    sor4onnx \
    --input_onnx_file_path 12_nms_score_gather_nd.onnx \
    --old_new "serving_default_input_2" "gn_selected_indices" \
    --output_onnx_file_path 12_nms_score_gather_nd.onnx \
    --mode inputs

    sor4onnx \
    --input_onnx_file_path 12_nms_score_gather_nd.onnx \
    --old_new "PartitionedCall" "final_scores" \
    --output_onnx_file_path 12_nms_score_gather_nd.onnx \
    --mode outputs

    python make_input_output_shape_update.py \
    --input_onnx_file_path 12_nms_score_gather_nd.onnx \
    --output_onnx_file_path 12_nms_score_gather_nd.onnx \
    --input_names gn_scores \
    --input_names gn_selected_indices \
    --input_shapes ${BATCHES} ${CLASSES} ${BOXES} \
    --input_shapes N 3 \
    --output_names final_scores \
    --output_shapes N 1

    onnxsim 12_nms_score_gather_nd.onnx 12_nms_score_gather_nd.onnx
    onnxsim 12_nms_score_gather_nd.onnx 12_nms_score_gather_nd.onnx


    ################################################### NonMaxSuppression + Score GatherND
    snc4onnx \
    --input_onnx_file_paths 09_nms_retinaface_${BOXES}.onnx 12_nms_score_gather_nd.onnx \
    --srcop_destop scores gn_scores workaround_mul_out gn_selected_indices \
    --disable_onnxsim \
    --output_onnx_file_path 09_nms_retinaface_${BOXES}_nd.onnx


    ################################################### Final Batch Nums
    python make_final_batch_nums_final_class_nums_final_box_nums.py


    ################################################### Boxes GatherND
    python make_box_gather_nd.py

    python -m tf2onnx.convert \
    --opset ${OPSET} \
    --tflite saved_model_postprocess/nms_box_gather_nd.tflite \
    --output 14_nms_box_gather_nd.onnx

    sor4onnx \
    --input_onnx_file_path 14_nms_box_gather_nd.onnx \
    --old_new ":0" "" \
    --search_mode "suffix_match" \
    --output_onnx_file_path 14_nms_box_gather_nd.onnx

    sor4onnx \
    --input_onnx_file_path 14_nms_box_gather_nd.onnx \
    --old_new "serving_default_input_1" "gn_boxes" \
    --output_onnx_file_path 14_nms_box_gather_nd.onnx \
    --mode inputs

    sor4onnx \
    --input_onnx_file_path 14_nms_box_gather_nd.onnx \
    --old_new "serving_default_input_2" "gn_box_selected_indices" \
    --output_onnx_file_path 14_nms_box_gather_nd.onnx \
    --mode inputs

    sor4onnx \
    --input_onnx_file_path 14_nms_box_gather_nd.onnx \
    --old_new "PartitionedCall" "final_boxes" \
    --output_onnx_file_path 14_nms_box_gather_nd.onnx \
    --mode outputs

    python make_input_output_shape_update.py \
    --input_onnx_file_path 14_nms_box_gather_nd.onnx \
    --output_onnx_file_path 14_nms_box_gather_nd.onnx \
    --input_names gn_boxes \
    --input_names gn_box_selected_indices \
    --input_shapes ${BATCHES} ${BOXES} 4 \
    --input_shapes N 2 \
    --output_names final_boxes \
    --output_shapes N 4

    onnxsim 14_nms_box_gather_nd.onnx 14_nms_box_gather_nd.onnx
    onnxsim 14_nms_box_gather_nd.onnx 14_nms_box_gather_nd.onnx



    ################################################### Landms GatherND
    python make_landms_gather_nd.py

    python -m tf2onnx.convert \
    --opset ${OPSET} \
    --tflite saved_model_postprocess/nms_landms_gather_nd.tflite \
    --output 14_nms_landms_gather_nd.onnx

    sor4onnx \
    --input_onnx_file_path 14_nms_landms_gather_nd.onnx \
    --old_new ":0" "" \
    --search_mode "suffix_match" \
    --output_onnx_file_path 14_nms_landms_gather_nd.onnx

    sor4onnx \
    --input_onnx_file_path 14_nms_landms_gather_nd.onnx \
    --old_new "serving_default_input_1" "gn_landms" \
    --output_onnx_file_path 14_nms_landms_gather_nd.onnx \
    --mode inputs

    sor4onnx \
    --input_onnx_file_path 14_nms_landms_gather_nd.onnx \
    --old_new "serving_default_input_2" "gn_landms_selected_indices" \
    --output_onnx_file_path 14_nms_landms_gather_nd.onnx \
    --mode inputs

    sor4onnx \
    --input_onnx_file_path 14_nms_landms_gather_nd.onnx \
    --old_new "PartitionedCall" "final_landms" \
    --output_onnx_file_path 14_nms_landms_gather_nd.onnx \
    --mode outputs

    python make_input_output_shape_update.py \
    --input_onnx_file_path 14_nms_landms_gather_nd.onnx \
    --output_onnx_file_path 14_nms_landms_gather_nd.onnx \
    --input_names gn_landms \
    --input_names gn_landms_selected_indices \
    --input_shapes ${BATCHES} ${BOXES} 10 \
    --input_shapes N 2 \
    --output_names final_landms \
    --output_shapes N 10

    onnxsim 14_nms_landms_gather_nd.onnx 14_nms_landms_gather_nd.onnx
    onnxsim 14_nms_landms_gather_nd.onnx 14_nms_landms_gather_nd.onnx


    ################################################### nms_retinaface_xxx_nd + nms_final_batch_nums_final_class_nums_final_box_nums
    snc4onnx \
    --input_onnx_file_paths 09_nms_retinaface_${BOXES}_nd.onnx 13_nms_final_batch_nums_final_class_nums_final_box_nums.onnx \
    --srcop_destop workaround_mul_out bc_input \
    --op_prefixes_after_merging main01 sub01 \
    --disable_onnxsim \
    --output_onnx_file_path 15_nms_retinaface_${BOXES}_split.onnx



    ################################################### nms_retinaface_${BOXES}_split + nms_box_gather_nd
    snc4onnx \
    --input_onnx_file_paths 15_nms_retinaface_${BOXES}_split.onnx 14_nms_box_gather_nd.onnx \
    --srcop_destop x1y1x2y2 gn_boxes final_box_nums gn_box_selected_indices \
    --disable_onnxsim \
    --output_onnx_file_path 16_nms_retinaface_${BOXES}_merged.onnx

    sor4onnx \
    --input_onnx_file_path 16_nms_retinaface_${BOXES}_merged.onnx \
    --old_new "sub01_/Gather" "nms_gather" \
    --output_onnx_file_path 16_nms_retinaface_${BOXES}_merged.onnx \
    --mode full \
    --search_mode prefix_match

    sor4onnx \
    --input_onnx_file_path 16_nms_retinaface_${BOXES}_merged.onnx \
    --old_new "PartitionedCall" "nms_gathernd2" \
    --output_onnx_file_path 16_nms_retinaface_${BOXES}_merged.onnx \
    --mode full \
    --search_mode prefix_match


    ################################################### nms_retinaface_${BOXES}_split + nms_landms_gather_nd
    snc4onnx \
    --input_onnx_file_paths 16_nms_retinaface_${BOXES}_merged.onnx 14_nms_landms_gather_nd.onnx \
    --srcop_destop final_box_nums gn_landms_selected_indices \
    --disable_onnxsim \
    --output_onnx_file_path 16_nms_retinaface_${BOXES}_merged.onnx

    sor4onnx \
    --input_onnx_file_path 16_nms_retinaface_${BOXES}_merged.onnx \
    --old_new "PartitionedCall" "nms_gathernd3" \
    --output_onnx_file_path 16_nms_retinaface_${BOXES}_merged.onnx \
    --mode full \
    --search_mode prefix_match


    ################################################### nms output merge
    python make_nms_outputs_merge.py

    onnxsim 17_nms_batchno_classid_x1y1x2y2_landms_cat.onnx 17_nms_batchno_classid_x1y1x2y2_landms_cat.onnx


    ################################################### merge
    snc4onnx \
    --input_onnx_file_paths 16_nms_retinaface_${BOXES}_merged.onnx 17_nms_batchno_classid_x1y1x2y2_landms_cat.onnx \
    --srcop_destop final_batch_nums cat_batch final_class_nums cat_classid final_scores cat_score final_boxes cat_x1y1x2y2 final_landms cat_landms \
    --disable_onnxsim \
    --output_onnx_file_path 18_nms_retinaface_${BOXES}.onnx


    ################################################### Extracted only after NMS
    sne4onnx \
    --input_onnx_file_path 18_nms_retinaface_${BOXES}.onnx \
    --input_op_names x1y1x2y2 main01_y1x1y2x2 main01_scores gn_landms \
    --output_op_names batchno_classid_score_x1y1x2y2_landms \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx

    sor4onnx \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --old_new "main01_" "nms_" \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --mode full \
    --search_mode prefix_match

    sor4onnx \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --old_new "x1y1x2y2" "nms_x1y1x2y2" \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --mode full \
    --search_mode prefix_match

    sor4onnx \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --old_new "gn_landms" "nms_landms" \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --mode full \
    --search_mode prefix_match

    sor4onnx \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --old_new "nms_nonmaxsuppression11" "nms_nonmaxsuppression" \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --mode full \
    --search_mode prefix_match

    sor4onnx \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --old_new "nms_mul11" "nms_mul" \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --mode full \
    --search_mode prefix_match

    sor4onnx \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --old_new "nms_02_/Constant_3_output_0" "nms_max_output_boxes_per_class" \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --mode full \
    --search_mode prefix_match

    sor4onnx \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --old_new "nms_model/tf.compat.v1.gather_nd/GatherNd" "nms_gathernd1" \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --mode full \
    --search_mode prefix_match

    sor4onnx \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --old_new "nms_model/tf.__operators__.getitem/strided_slice1" "nms_reshape1" \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --mode full \
    --search_mode prefix_match

    sor4onnx \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --old_new "sub01_/Slice" "nms_slice1" \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --mode full \
    --search_mode prefix_match

    sor4onnx \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --old_new "nms_slice1_1" "nms_slice2" \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --mode full \
    --search_mode prefix_match

    sor4onnx \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --old_new "nms_PartitionedCall" "nms_slice3" \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --mode full \
    --search_mode prefix_match

    sor4onnx \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --old_new "/Concat" "nms_concat1" \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --mode full \
    --search_mode prefix_match

    sor4onnx \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --old_new "sub01_/Cast_1" "nms_cast2" \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --mode full \
    --search_mode prefix_match

    sor4onnx \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --old_new "sub01_/Cast" "nms_cast1" \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --mode full \
    --search_mode prefix_match



    ################################################### NMS
    onnx2json \
    --input_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx \
    --output_json_path 19_nms_retinaface_1x${BOXES}x4.json \
    --json_indent 2

    sed -i -e 's/"dimParam": "1"/"dimValue": "1"/g' 19_nms_retinaface_1x${BOXES}x4.json
    sed -i -e "s/\"dimParam\": \"${BOXES}\"/\"dimValue\": \"${BOXES}\"/g" 19_nms_retinaface_1x${BOXES}x4.json
    sed -i -e 's/"dimParam": "10"/"dimValue": "10"/g' 19_nms_retinaface_1x${BOXES}x4.json

    json2onnx \
    --input_json_path 19_nms_retinaface_1x${BOXES}x4.json \
    --output_onnx_file_path 19_nms_retinaface_1x${BOXES}x4.onnx

    rm 19_nms_retinaface_1x12600x4.json


    ################################################### pre-NMS
    python make_post_process.py \
    --image_height ${IMAGE_HEIGHT} \
    --image_width ${IMAGE_WIDTH} \
    --opset ${OPSET}


    ################################################### pre-NMS + NMS
    snc4onnx \
    --input_onnx_file_paths 19_nms_retinaface_1x${BOXES}x4.onnx 27_post_process_prenms_1x${IMAGE_HEIGHT}x${IMAGE_WIDTH}_${OPSET}.onnx \
    --output_onnx_file_path retinaface_post_process1x${IMAGE_HEIGHT}x${IMAGE_WIDTH}_${OPSET}.onnx \
    --srcop_destop prenms_output_boxes_xyxy nms_x1y1x2y2 prenms_output_boxes_yxyx nms_y1x1y2x2 prenms_output_scores nms_scores prenms_output_landms nms_landms \
    --disable_onnxsim

    sor4onnx \
    --input_onnx_file_path retinaface_post_process1x${IMAGE_HEIGHT}x${IMAGE_WIDTH}_${OPSET}.onnx \
    --old_new "/Concat" "post_concat" \
    --output_onnx_file_path retinaface_post_process1x${IMAGE_HEIGHT}x${IMAGE_WIDTH}_${OPSET}.onnx \
    --mode full \
    --search_mode exact_match

    sor4onnx \
    --input_onnx_file_path retinaface_post_process1x${IMAGE_HEIGHT}x${IMAGE_WIDTH}_${OPSET}.onnx \
    --old_new "/Concat_1_output_0" "post_concat_const" \
    --output_onnx_file_path retinaface_post_process1x${IMAGE_HEIGHT}x${IMAGE_WIDTH}_${OPSET}.onnx \
    --mode full \
    --search_mode exact_match


    ################################################### cleaning
    rm 0*_*.onnx
    rm 1*_*.onnx
    rm 1*_*.json
    rm 2*_*.onnx


    ################################################### RetinaFace + Post-Process merge
    snc4onnx \
    --input_onnx_file_paths retinaface_post_process1x${IMAGE_HEIGHT}x${IMAGE_WIDTH}_${OPSET}.onnx retinaface_mobilenet0.25_Final_${IMAGE_HEIGHT}x${IMAGE_WIDTH}.onnx \
    --output_onnx_file_path retinaface_mbn025_with_postprocess_${IMAGE_HEIGHT}x${IMAGE_WIDTH}_max${MAX_OUTPUT_BOXES_PER_CLASS}.onnx \
    --srcop_destop loc prenms_loc conf prenms_conf landms prenms_landms \
    --disable_onnxsim
done

