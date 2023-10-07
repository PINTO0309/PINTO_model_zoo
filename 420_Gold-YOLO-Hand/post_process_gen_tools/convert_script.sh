# pip install -U pip \
# && pip install onnxsim==0.4.33 \
# && pip install -U simple-onnx-processing-tools \
# && pip install -U onnx \
# && python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

MODEL_NAME=gold_yolo_n_hand
SUFFIX="0423_0.2295_1x3x"
OPSET=11
BATCHES=1
CLASSES=1

RESOLUTIONS=(
    "192 320 1260"
    "192 416 1638"
    "192 640 2520"
    "192 800 3150"
    "256 320 1680"
    "256 416 2184"
    "256 640 3360"
    "256 800 4200"
    "256 960 5040"
    "288 1280 7560"
    "288 480 2835"
    "288 640 3780"
    "288 800 4725"
    "288 960 5670"
    "320 320 2100"
    "384 1280 10080"
    "384 480 3780"
    "384 640 5040"
    "384 800 6300"
    "384 960 7560"
    "416 416 3549"
    "480 1280 12600"
    "480 640 6300"
    "480 800 7875"
    "480 960 9450"
    "512 512 5376"
    "512 640 6720"
    "512 896 9408"
    "544 1280 14280"
    "544 800 8925"
    "544 960 10710"
    "640 640 8400"
    "736 1280 19320"
)

for((i=0; i<${#RESOLUTIONS[@]}; i++))
do
    RESOLUTION=(`echo ${RESOLUTIONS[i]}`)
    H=${RESOLUTION[0]}
    W=${RESOLUTION[1]}
    BOXES=${RESOLUTION[2]}

    ################################################### Boxes + Scores
    python make_boxes_scores.py -o ${OPSET} -b ${BATCHES} -x ${BOXES} -c ${CLASSES}
    python make_cxcywh_y1x1y2x2.py -o ${OPSET} -b ${BATCHES} -x ${BOXES}

    sor4onnx \
    --input_onnx_file_path 01_boxes_scores_${BOXES}.onnx \
    --old_new "/Constant" "boxes_scores_Constant" \
    --mode full \
    --search_mode prefix_match \
    --output_onnx_file_path 01_boxes_scores_${BOXES}.onnx

    sor4onnx \
    --input_onnx_file_path 02_cxcywh_y1x1y2x2_${BOXES}.onnx \
    --old_new "/Constant" "cxcywh_y1x1y2x2_Constant" \
    --mode full \
    --search_mode prefix_match \
    --output_onnx_file_path 02_cxcywh_y1x1y2x2_${BOXES}.onnx

    sor4onnx \
    --input_onnx_file_path 02_cxcywh_y1x1y2x2_${BOXES}.onnx \
    --old_new "/Slice" "cxcywh_y1x1y2x2_Slice" \
    --mode full \
    --search_mode prefix_match \
    --output_onnx_file_path 02_cxcywh_y1x1y2x2_${BOXES}.onnx

    snc4onnx \
    --input_onnx_file_paths 01_boxes_scores_${BOXES}.onnx 02_cxcywh_y1x1y2x2_${BOXES}.onnx \
    --srcop_destop boxes_cxcywh cxcywh \
    --output_onnx_file_path 03_boxes_y1x1y2x2_scores_${BOXES}.onnx


    ################################################### NonMaxSuppression
    sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name max_output_boxes_per_class_const \
    --output_variables max_output_boxes_per_class int64 [1] \
    --attributes value int64 [20] \
    --output_onnx_file_path 04_Constant_max_output_boxes_per_class.onnx

    sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name iou_threshold_const \
    --output_variables iou_threshold float32 [1] \
    --attributes value float32 [0.05] \
    --output_onnx_file_path 05_Constant_iou_threshold.onnx

    sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name score_threshold_const \
    --output_variables score_threshold float32 [1] \
    --attributes value float32 [0.50] \
    --output_onnx_file_path 06_Constant_score_threshold.onnx


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
    --output_onnx_file_path 07_${OP}${OPSET}.onnx


    snc4onnx \
    --input_onnx_file_paths 04_Constant_max_output_boxes_per_class.onnx 07_${OP}${OPSET}.onnx \
    --srcop_destop max_output_boxes_per_class max_output_boxes_per_class_var \
    --output_onnx_file_path 07_${OP}${OPSET}.onnx

    snc4onnx \
    --input_onnx_file_paths 05_Constant_iou_threshold.onnx 07_${OP}${OPSET}.onnx \
    --srcop_destop iou_threshold iou_threshold_var \
    --output_onnx_file_path 07_${OP}${OPSET}.onnx

    snc4onnx \
    --input_onnx_file_paths 06_Constant_score_threshold.onnx 07_${OP}${OPSET}.onnx \
    --srcop_destop score_threshold score_threshold_var \
    --output_onnx_file_path 07_${OP}${OPSET}.onnx

    soc4onnx \
    --input_onnx_file_path 07_${OP}${OPSET}.onnx \
    --output_onnx_file_path 07_${OP}${OPSET}.onnx \
    --opset ${OPSET}


    ################################################### Boxes + Scores + NonMaxSuppression
    snc4onnx \
    --input_onnx_file_paths 03_boxes_y1x1y2x2_scores_${BOXES}.onnx 07_${OP}${OPSET}.onnx \
    --srcop_destop scores scores_var y1x1y2x2 boxes_var \
    --output_onnx_file_path 08_nms_${MODEL_NAME}_${BOXES}.onnx


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
    --output_onnx_file_path 09_${OP}${OPSET}_workaround.onnx

    ############ Myriad workaround Constant
    sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name workaround_mul_const_op \
    --output_variables workaround_mul_const int64 [1] \
    --attributes value int64 [1] \
    --output_onnx_file_path 10_Constant_workaround_mul.onnx

    ############ Myriad workaround Mul + Myriad workaround Constant
    snc4onnx \
    --input_onnx_file_paths 10_Constant_workaround_mul.onnx 09_${OP}${OPSET}_workaround.onnx \
    --srcop_destop workaround_mul_const workaround_mul_b \
    --output_onnx_file_path 09_${OP}${OPSET}_workaround.onnx



    ################################################### NonMaxSuppression + Myriad workaround Mul
    snc4onnx \
    --input_onnx_file_paths 08_nms_${MODEL_NAME}_${BOXES}.onnx 09_${OP}${OPSET}_workaround.onnx \
    --srcop_destop selected_indices workaround_mul_a \
    --output_onnx_file_path 11_nms_${MODEL_NAME}_${BOXES}.onnx \
    --disable_onnxsim

    ################################################### N batch NMS
    sbi4onnx \
    --input_onnx_file_path 11_nms_${MODEL_NAME}_${BOXES}.onnx \
    --output_onnx_file_path 12_nms_${MODEL_NAME}_${BOXES}_batch.onnx \
    --initialization_character_string batch

    sio4onnx \
    --input_onnx_file_path 12_nms_${MODEL_NAME}_${BOXES}_batch.onnx \
    --output_onnx_file_path 12_nms_${MODEL_NAME}_${BOXES}_batch.onnx \
    --input_names "predictions" \
    --input_shapes "batch" ${BOXES} $((CLASSES+5)) \
    --output_names "x1y1x2y2" \
    --output_names "workaround_mul_out" \
    --output_shapes "batch" ${BOXES} 4 \
    --output_shapes "N" 3




    ################################################### Score GatherND
    python make_score_gather_nd.py -b ${BATCHES} -x ${BOXES} -c ${CLASSES}

    python -m tf2onnx.convert \
    --opset ${OPSET} \
    --tflite saved_model_postprocess/nms_score_gather_nd.tflite \
    --output 13_nms_score_gather_nd.onnx

    sor4onnx \
    --input_onnx_file_path 13_nms_score_gather_nd.onnx \
    --old_new ":0" "" \
    --mode full \
    --search_mode partial_match \
    --output_onnx_file_path 13_nms_score_gather_nd.onnx

    sor4onnx \
    --input_onnx_file_path 13_nms_score_gather_nd.onnx \
    --old_new "serving_default_input_1" "gn_scores" \
    --output_onnx_file_path 13_nms_score_gather_nd.onnx \
    --mode inputs

    sor4onnx \
    --input_onnx_file_path 13_nms_score_gather_nd.onnx \
    --old_new "serving_default_input_2" "gn_selected_indices" \
    --output_onnx_file_path 13_nms_score_gather_nd.onnx \
    --mode inputs

    sor4onnx \
    --input_onnx_file_path 13_nms_score_gather_nd.onnx \
    --old_new "PartitionedCall" "final_scores" \
    --output_onnx_file_path 13_nms_score_gather_nd.onnx \
    --mode outputs

    python make_input_output_shape_update.py \
    --input_onnx_file_path 13_nms_score_gather_nd.onnx \
    --output_onnx_file_path 13_nms_score_gather_nd.onnx \
    --input_names gn_scores \
    --input_names gn_selected_indices \
    --input_shapes ${BATCHES} ${CLASSES} ${BOXES} \
    --input_shapes N 3 \
    --output_names final_scores \
    --output_shapes N 1

    onnxsim 13_nms_score_gather_nd.onnx 13_nms_score_gather_nd.onnx
    onnxsim 13_nms_score_gather_nd.onnx 13_nms_score_gather_nd.onnx

    sio4onnx \
    --input_onnx_file_path 13_nms_score_gather_nd.onnx \
    --output_onnx_file_path 14_nms_score_gather_nd_batch.onnx \
    --input_names "gn_scores" \
    --input_names "gn_selected_indices" \
    --input_shapes "batch" ${CLASSES} ${BOXES} \
    --input_shapes "N" 3 \
    --output_names "final_scores" \
    --output_shapes "N" 1


    ################################################### NonMaxSuppression + Score GatherND
    snc4onnx \
    --input_onnx_file_paths 11_nms_${MODEL_NAME}_${BOXES}.onnx 13_nms_score_gather_nd.onnx \
    --srcop_destop scores gn_scores workaround_mul_out gn_selected_indices \
    --output_onnx_file_path 15_nms_${MODEL_NAME}_${BOXES}_nd.onnx

    onnxsim 15_nms_${MODEL_NAME}_${BOXES}_nd.onnx 15_nms_${MODEL_NAME}_${BOXES}_nd.onnx
    onnxsim 15_nms_${MODEL_NAME}_${BOXES}_nd.onnx 15_nms_${MODEL_NAME}_${BOXES}_nd.onnx


    snc4onnx \
    --input_onnx_file_paths 12_nms_${MODEL_NAME}_${BOXES}_batch.onnx 14_nms_score_gather_nd_batch.onnx \
    --srcop_destop scores gn_scores workaround_mul_out gn_selected_indices \
    --output_onnx_file_path 16_nms_${MODEL_NAME}_${BOXES}_nd_batch.onnx

    onnxsim 16_nms_${MODEL_NAME}_${BOXES}_nd_batch.onnx 16_nms_${MODEL_NAME}_${BOXES}_nd_batch.onnx
    onnxsim 16_nms_${MODEL_NAME}_${BOXES}_nd_batch.onnx 16_nms_${MODEL_NAME}_${BOXES}_nd_batch.onnx







    ################################################### Final Batch Nums
    python make_final_batch_nums_final_class_nums_final_box_nums.py


    ################################################### Boxes GatherND
    python make_box_gather_nd.py

    python -m tf2onnx.convert \
    --opset ${OPSET} \
    --tflite saved_model_postprocess/nms_box_gather_nd.tflite \
    --output 18_nms_box_gather_nd.onnx

    sor4onnx \
    --input_onnx_file_path 18_nms_box_gather_nd.onnx \
    --old_new ":0" "" \
    --mode full \
    --search_mode partial_match \
    --output_onnx_file_path 18_nms_box_gather_nd.onnx

    sor4onnx \
    --input_onnx_file_path 18_nms_box_gather_nd.onnx \
    --old_new "serving_default_input_1" "gn_boxes" \
    --output_onnx_file_path 18_nms_box_gather_nd.onnx \
    --mode inputs

    sor4onnx \
    --input_onnx_file_path 18_nms_box_gather_nd.onnx \
    --old_new "serving_default_input_2" "gn_box_selected_indices" \
    --output_onnx_file_path 18_nms_box_gather_nd.onnx \
    --mode inputs

    sor4onnx \
    --input_onnx_file_path 18_nms_box_gather_nd.onnx \
    --old_new "PartitionedCall" "final_boxes" \
    --output_onnx_file_path 18_nms_box_gather_nd.onnx \
    --mode outputs

    python make_input_output_shape_update.py \
    --input_onnx_file_path 18_nms_box_gather_nd.onnx \
    --output_onnx_file_path 18_nms_box_gather_nd.onnx \
    --input_names gn_boxes \
    --input_names gn_box_selected_indices \
    --input_shapes ${BATCHES} ${BOXES} 4 \
    --input_shapes N 2 \
    --output_names final_boxes \
    --output_shapes N 4

    onnxsim 18_nms_box_gather_nd.onnx 18_nms_box_gather_nd.onnx
    onnxsim 18_nms_box_gather_nd.onnx 18_nms_box_gather_nd.onnx

    sio4onnx \
    --input_onnx_file_path 18_nms_box_gather_nd.onnx \
    --output_onnx_file_path 19_nms_box_gather_nd_batch.onnx \
    --input_names "gn_boxes" \
    --input_names "gn_box_selected_indices" \
    --input_shapes "batch" ${BOXES} 4 \
    --input_shapes "N" 2 \
    --output_names "final_boxes" \
    --output_shapes "N" 4


    ################################################### nms_${MODEL_NAME}_xxx_nd + nms_final_batch_nums_final_class_nums_final_box_nums
    snc4onnx \
    --input_onnx_file_paths 15_nms_${MODEL_NAME}_${BOXES}_nd.onnx 17_nms_final_batch_nums_final_class_nums_final_box_nums.onnx \
    --srcop_destop selected_indices bc_input \
    --op_prefixes_after_merging main01 sub01 \
    --output_onnx_file_path 20_nms_${MODEL_NAME}_${BOXES}_split.onnx

    snc4onnx \
    --input_onnx_file_paths 16_nms_${MODEL_NAME}_${BOXES}_nd_batch.onnx 17_nms_final_batch_nums_final_class_nums_final_box_nums.onnx \
    --srcop_destop workaround_mul_out bc_input \
    --op_prefixes_after_merging main01 sub01 \
    --output_onnx_file_path 21_nms_${MODEL_NAME}_${BOXES}_split_batch.onnx



    ################################################### nms_${MODEL_NAME}_${BOXES}_split + nms_box_gather_nd
    snc4onnx \
    --input_onnx_file_paths 20_nms_${MODEL_NAME}_${BOXES}_split.onnx 18_nms_box_gather_nd.onnx \
    --srcop_destop x1y1x2y2 gn_boxes final_box_nums gn_box_selected_indices \
    --output_onnx_file_path 22_nms_${MODEL_NAME}_${BOXES}_merged.onnx

    onnxsim 22_nms_${MODEL_NAME}_${BOXES}_merged.onnx 22_nms_${MODEL_NAME}_${BOXES}_merged.onnx
    onnxsim 22_nms_${MODEL_NAME}_${BOXES}_merged.onnx 22_nms_${MODEL_NAME}_${BOXES}_merged.onnx


    snc4onnx \
    --input_onnx_file_paths 21_nms_${MODEL_NAME}_${BOXES}_split_batch.onnx 19_nms_box_gather_nd_batch.onnx \
    --srcop_destop x1y1x2y2 gn_boxes final_box_nums gn_box_selected_indices \
    --output_onnx_file_path 23_nms_${MODEL_NAME}_${BOXES}_merged_batch.onnx

    onnxsim 23_nms_${MODEL_NAME}_${BOXES}_merged_batch.onnx 23_nms_${MODEL_NAME}_${BOXES}_merged_batch.onnx
    onnxsim 23_nms_${MODEL_NAME}_${BOXES}_merged_batch.onnx 23_nms_${MODEL_NAME}_${BOXES}_merged_batch.onnx





    ################################################### nms output op name Cleaning
    sor4onnx \
    --input_onnx_file_path 22_nms_${MODEL_NAME}_${BOXES}_merged.onnx \
    --old_new "main01_final_scores" "final_scores" \
    --output_onnx_file_path 22_nms_${MODEL_NAME}_${BOXES}_merged.onnx \
    --mode outputs

    sor4onnx \
    --input_onnx_file_path 22_nms_${MODEL_NAME}_${BOXES}_merged.onnx \
    --old_new "sub01_final_batch_nums" "final_batch_nums" \
    --output_onnx_file_path 22_nms_${MODEL_NAME}_${BOXES}_merged.onnx \
    --mode outputs

    sor4onnx \
    --input_onnx_file_path 22_nms_${MODEL_NAME}_${BOXES}_merged.onnx \
    --old_new "sub01_final_class_nums" "final_class_nums" \
    --output_onnx_file_path 22_nms_${MODEL_NAME}_${BOXES}_merged.onnx \
    --mode outputs


    sor4onnx \
    --input_onnx_file_path 23_nms_${MODEL_NAME}_${BOXES}_merged_batch.onnx \
    --old_new "main01_final_scores" "final_scores" \
    --output_onnx_file_path 23_nms_${MODEL_NAME}_${BOXES}_merged_batch.onnx \
    --mode outputs

    sor4onnx \
    --input_onnx_file_path 23_nms_${MODEL_NAME}_${BOXES}_merged_batch.onnx \
    --old_new "sub01_final_batch_nums" "final_batch_nums" \
    --output_onnx_file_path 23_nms_${MODEL_NAME}_${BOXES}_merged_batch.onnx \
    --mode outputs

    sor4onnx \
    --input_onnx_file_path 23_nms_${MODEL_NAME}_${BOXES}_merged_batch.onnx \
    --old_new "sub01_final_class_nums" "final_class_nums" \
    --output_onnx_file_path 23_nms_${MODEL_NAME}_${BOXES}_merged_batch.onnx \
    --mode outputs







    ################################################### nms output merge
    python make_nms_outputs_merge.py

    onnxsim 24_nms_batchno_classid_x1y1x2y2_score_cat.onnx 24_nms_batchno_classid_x1y1x2y2_score_cat.onnx


    ################################################### merge
    snc4onnx \
    --input_onnx_file_paths 22_nms_${MODEL_NAME}_${BOXES}_merged.onnx 24_nms_batchno_classid_x1y1x2y2_score_cat.onnx \
    --srcop_destop final_batch_nums cat_batch final_class_nums cat_classid final_boxes cat_x1y1x2y2 final_scores cat_score \
    --output_onnx_file_path 30_nms_${MODEL_NAME}_${BOXES}.onnx

    sor4onnx \
    --input_onnx_file_path 30_nms_${MODEL_NAME}_${BOXES}.onnx \
    --old_new "final_scores" "score" \
    --output_onnx_file_path 30_nms_${MODEL_NAME}_${BOXES}.onnx \
    --mode outputs



    ################################################### merge
    snc4onnx \
    --input_onnx_file_paths 23_nms_${MODEL_NAME}_${BOXES}_merged_batch.onnx 24_nms_batchno_classid_x1y1x2y2_score_cat.onnx \
    --srcop_destop final_batch_nums cat_batch final_class_nums cat_classid final_boxes cat_x1y1x2y2 final_scores cat_score \
    --output_onnx_file_path 31_nms_${MODEL_NAME}_N_${BOXES}.onnx

    sor4onnx \
    --input_onnx_file_path 31_nms_${MODEL_NAME}_N_${BOXES}.onnx \
    --old_new "final_scores" "score" \
    --output_onnx_file_path 31_nms_${MODEL_NAME}_N_${BOXES}.onnx \
    --mode outputs

    # ################################################### Cleaning
    rm 0*.onnx
    rm 1*.onnx
    rm 2*.onnx


    ################################################### ${MODEL_NAME} + Post-Process
    snc4onnx \
    --input_onnx_file_paths ${MODEL_NAME}_${SUFFIX}${H}x${W}.onnx 30_nms_${MODEL_NAME}_${BOXES}.onnx \
    --srcop_destop output predictions \
    --output_onnx_file_path ${MODEL_NAME}_post_${SUFFIX}${H}x${W}.onnx
    onnxsim ${MODEL_NAME}_post_${SUFFIX}${H}x${W}.onnx ${MODEL_NAME}_post_${SUFFIX}${H}x${W}.onnx
    onnxsim ${MODEL_NAME}_post_${SUFFIX}${H}x${W}.onnx ${MODEL_NAME}_post_${SUFFIX}${H}x${W}.onnx
done
