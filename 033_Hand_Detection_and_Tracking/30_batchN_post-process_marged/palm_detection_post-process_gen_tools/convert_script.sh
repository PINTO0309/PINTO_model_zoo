#!/bin/bash

OPSET=11
BATCHES=1
BOXES_LIST=(
    "896"
    "2016"
)
CLASSES=1

for((i=0; i<${#BOXES_LIST[@]}; i++))
do
    BOXES=(`echo ${BOXES_LIST[i]}`)

    ################################################### NonMaxSuppression
    sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name max_output_boxes_per_class_const \
    --output_variables max_output_boxes_per_class int64 [1] \
    --attributes value int64 [${BOXES}] \
    --output_onnx_file_path Constant_max_output_boxes_per_class.onnx

    sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name iou_threshold_const \
    --output_variables iou_threshold float32 [1] \
    --attributes value float32 [0.3] \
    --output_onnx_file_path Constant_iou_threshold.onnx

    sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name score_threshold_const \
    --output_variables score_threshold float32 [1] \
    --attributes value float32 [0.6] \
    --output_onnx_file_path Constant_score_threshold.onnx

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
    --output_onnx_file_path ${OP}${OPSET}_${BOXES}.onnx

    snc4onnx \
    --input_onnx_file_paths Constant_max_output_boxes_per_class.onnx ${OP}${OPSET}_${BOXES}.onnx \
    --srcop_destop max_output_boxes_per_class max_output_boxes_per_class_var \
    --output_onnx_file_path ${OP}${OPSET}_${BOXES}.onnx

    snc4onnx \
    --input_onnx_file_paths Constant_iou_threshold.onnx ${OP}${OPSET}_${BOXES}.onnx \
    --srcop_destop iou_threshold iou_threshold_var \
    --output_onnx_file_path ${OP}${OPSET}_${BOXES}.onnx

    snc4onnx \
    --input_onnx_file_paths Constant_score_threshold.onnx ${OP}${OPSET}_${BOXES}.onnx \
    --srcop_destop score_threshold score_threshold_var \
    --output_onnx_file_path ${OP}${OPSET}_${BOXES}.onnx

    soc4onnx \
    --input_onnx_file_path ${OP}${OPSET}_${BOXES}.onnx \
    --output_onnx_file_path ${OP}${OPSET}_${BOXES}.onnx \
    --opset ${OPSET}

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
    --output_onnx_file_path ${OP}${OPSET}_workaround.onnx

    ############ Myriad workaround Constant
    sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name workaround_mul_const_op \
    --output_variables workaround_mul_const int64 [1] \
    --attributes value int64 [1] \
    --output_onnx_file_path Constant_workaround_mul.onnx

    ############ Myriad workaround Mul + Myriad workaround Constant
    snc4onnx \
    --input_onnx_file_paths Constant_workaround_mul.onnx ${OP}${OPSET}_workaround.onnx \
    --srcop_destop workaround_mul_const workaround_mul_b \
    --output_onnx_file_path ${OP}${OPSET}_workaround.onnx

    ################################################### N batch NonMaxSuppression
    sbi4onnx \
    -if NonMaxSuppression${OPSET}_${BOXES}.onnx \
    -of NonMaxSuppression${OPSET}_N_${BOXES}.onnx \
    -ics batch

    sio4onnx \
    -if NonMaxSuppression${OPSET}_N_${BOXES}.onnx \
    -of NonMaxSuppression${OPSET}_N_${BOXES}.onnx \
    -i boxes_var \
    -i scores_var \
    -is "batch" ${BOXES} 4 \
    -is "batch" ${CLASSES} ${BOXES} \
    -o selected_indices \
    -os "N" 3

    ################################################### NonMaxSuppression + Myriad workaround Mul
    snc4onnx \
    --input_onnx_file_paths NonMaxSuppression${OPSET}_${BOXES}.onnx ${OP}${OPSET}_workaround.onnx \
    --srcop_destop selected_indices workaround_mul_a \
    --output_onnx_file_path NonMaxSuppression${OPSET}_${BOXES}.onnx

    snc4onnx \
    --input_onnx_file_paths NonMaxSuppression${OPSET}_N_${BOXES}.onnx ${OP}${OPSET}_workaround.onnx \
    --srcop_destop selected_indices workaround_mul_a \
    --output_onnx_file_path NonMaxSuppression${OPSET}_N_${BOXES}.onnx

    ################################################### Cleaning
    rm Constant_iou_threshold.onnx
    rm Constant_max_output_boxes_per_class.onnx
    rm Constant_score_threshold.onnx
    rm Constant_workaround_mul.onnx
    rm Mul11_workaround.onnx
done


snc4onnx \
--input_onnx_file_paths PDPostProcessing_reg_class_1x3x128.onnx NonMaxSuppression11_896.onnx \
--srcop_destop bb_y1x1y2x2 boxes_var scores scores_var \
--output_onnx_file_path PDPostProcessing_reg_class_1x3x128x128.onnx

snc4onnx \
--input_onnx_file_paths PDPostProcessing_reg_class_1x3x192.onnx NonMaxSuppression11_2016.onnx \
--srcop_destop bb_y1x1y2x2 boxes_var scores scores_var \
--output_onnx_file_path PDPostProcessing_reg_class_1x3x192x192.onnx

snc4onnx \
--input_onnx_file_paths PDPostProcessing_reg_class_Nx3x128.onnx NonMaxSuppression11_N_896.onnx \
--srcop_destop bb_y1x1y2x2 boxes_var scores scores_var \
--output_onnx_file_path PDPostProcessing_reg_class_Nx3x128x128.onnx

snc4onnx \
--input_onnx_file_paths PDPostProcessing_reg_class_Nx3x192.onnx NonMaxSuppression11_N_2016.onnx \
--srcop_destop bb_y1x1y2x2 boxes_var scores scores_var \
--output_onnx_file_path PDPostProcessing_reg_class_Nx3x192x192.onnx




################################################### Score GatherND
OPSET=11
BATCHES=1
BOXES_LIST=(
    "896"
    "2016"
)
CLASSES=1

for((i=0; i<${#BOXES_LIST[@]}; i++))
do
    BOXES=(`echo ${BOXES_LIST[i]}`)
    python make_score_gather_nd.py -b ${BATCHES} -x ${BOXES} -c ${CLASSES}

    python -m tf2onnx.convert \
    --opset ${OPSET} \
    --tflite saved_model_postprocess/nms_score_gather_nd_${BOXES}.tflite \
    --output nms_score_gather_nd_${BOXES}.onnx

    sor4onnx \
    --input_onnx_file_path nms_score_gather_nd_${BOXES}.onnx \
    --old_new ":0" "" \
    --output_onnx_file_path nms_score_gather_nd_${BOXES}.onnx

    sor4onnx \
    --input_onnx_file_path nms_score_gather_nd_${BOXES}.onnx \
    --old_new "serving_default_input_1" "gn_scores" \
    --output_onnx_file_path nms_score_gather_nd_${BOXES}.onnx \
    --mode inputs

    sor4onnx \
    --input_onnx_file_path nms_score_gather_nd_${BOXES}.onnx \
    --old_new "serving_default_input_2" "gn_selected_indices" \
    --output_onnx_file_path nms_score_gather_nd_${BOXES}.onnx \
    --mode inputs

    sor4onnx \
    --input_onnx_file_path nms_score_gather_nd_${BOXES}.onnx \
    --old_new "PartitionedCall" "final_scores" \
    --output_onnx_file_path nms_score_gather_nd_${BOXES}.onnx \
    --mode outputs

    python make_input_output_shape_update.py \
    --input_onnx_file_path nms_score_gather_nd_${BOXES}.onnx \
    --output_onnx_file_path nms_score_gather_nd_${BOXES}.onnx \
    --input_names gn_scores \
    --input_names gn_selected_indices \
    --input_shapes ${BATCHES} ${CLASSES} ${BOXES} \
    --input_shapes N 3 \
    --output_names final_scores \
    --output_shapes N 1

    onnxsim nms_score_gather_nd_${BOXES}.onnx nms_score_gather_nd_${BOXES}.onnx
    onnxsim nms_score_gather_nd_${BOXES}.onnx nms_score_gather_nd_${BOXES}.onnx

    sio4onnx \
    -if nms_score_gather_nd_${BOXES}.onnx \
    -of nms_score_gather_nd_N_${BOXES}.onnx \
    -i gn_scores \
    -i gn_selected_indices \
    -is "batch" ${CLASSES} ${BOXES} \
    -is "N" 3 \
    -o final_scores \
    -os "N" 1
done
rm -rf saved_model_postprocess


################################################### Boxes GatherND
OPSET=11
BATCHES=1
BOXES_LIST=(
    "896"
    "2016"
)
CLASSES=1

for((i=0; i<${#BOXES_LIST[@]}; i++))
do
    BOXES=(`echo ${BOXES_LIST[i]}`)
    python make_box_gather_nd.py -b ${BATCHES} -x ${BOXES}

    python -m tf2onnx.convert \
    --opset ${OPSET} \
    --tflite saved_model_postprocess/nms_box_gather_nd_${BOXES}.tflite \
    --output nms_box_gather_nd_${BOXES}.onnx

    sor4onnx \
    --input_onnx_file_path nms_box_gather_nd_${BOXES}.onnx \
    --old_new ":0" "" \
    --output_onnx_file_path nms_box_gather_nd_${BOXES}.onnx

    sor4onnx \
    --input_onnx_file_path nms_box_gather_nd_${BOXES}.onnx \
    --old_new "serving_default_input_1" "gn_boxes" \
    --output_onnx_file_path nms_box_gather_nd_${BOXES}.onnx \
    --mode inputs

    sor4onnx \
    --input_onnx_file_path nms_box_gather_nd_${BOXES}.onnx \
    --old_new "serving_default_input_2" "gn_box_selected_indices" \
    --output_onnx_file_path nms_box_gather_nd_${BOXES}.onnx \
    --mode inputs

    sor4onnx \
    --input_onnx_file_path nms_box_gather_nd_${BOXES}.onnx \
    --old_new "PartitionedCall" "final_boxes" \
    --output_onnx_file_path nms_box_gather_nd_${BOXES}.onnx \
    --mode outputs

    python make_input_output_shape_update.py \
    --input_onnx_file_path nms_box_gather_nd_${BOXES}.onnx \
    --output_onnx_file_path nms_box_gather_nd_${BOXES}.onnx \
    --input_names gn_boxes \
    --input_names gn_box_selected_indices \
    --input_shapes ${BATCHES} ${BOXES} 7 \
    --input_shapes N 2 \
    --output_names final_boxes \
    --output_shapes N 7

    onnxsim nms_box_gather_nd_${BOXES}.onnx nms_box_gather_nd_${BOXES}.onnx
    onnxsim nms_box_gather_nd_${BOXES}.onnx nms_box_gather_nd_${BOXES}.onnx

    sio4onnx \
    -if nms_box_gather_nd_${BOXES}.onnx \
    -of nms_box_gather_nd_N_${BOXES}.onnx \
    -i gn_boxes \
    -i gn_box_selected_indices \
    -is "batch" ${BOXES} 7 \
    -is "N" 2 \
    -o final_boxes \
    -os "N" 7
done
rm -rf saved_model_postprocess


################################################### Final Batch Nums
python make_final_batch_nums_final_class_nums_final_box_nums.py





OPSET=11
BATCHES=1
BOXES_LIST=(
    "128 128 896"
    "192 192 2016"
)
CLASSES=1
for((i=0; i<${#BOXES_LIST[@]}; i++))
do
    HWBOXES=(`echo ${BOXES_LIST[i]}`)
    H=${HWBOXES[0]}
    W=${HWBOXES[1]}
    BOXES=${HWBOXES[2]}

    ################################################### NonMaxSuppression + nms_final_batch_nums_final_class_nums_final_box_nums
    snc4onnx \
    --input_onnx_file_paths PDPostProcessing_reg_class_1x3x${H}x${W}.onnx nms_final_batch_nums_final_class_nums_final_box_nums.onnx \
    --srcop_destop workaround_mul_out bc_input \
    --output_onnx_file_path PDPostProcessing_reg_class_1x3x${H}x${W}_split.onnx

    snc4onnx \
    --input_onnx_file_paths PDPostProcessing_reg_class_Nx3x${H}x${W}.onnx nms_final_batch_nums_final_class_nums_final_box_nums.onnx \
    --srcop_destop workaround_mul_out bc_input \
    --output_onnx_file_path PDPostProcessing_reg_class_Nx3x${H}x${W}_split.onnx

    ################################################### NonMaxSuppression_split + Boxes GatherND
    snc4onnx \
    --input_onnx_file_paths PDPostProcessing_reg_class_1x3x${H}x${W}_split.onnx nms_box_gather_nd_${BOXES}.onnx \
    --srcop_destop cxcyw_wristcenterxy_middlefingerxy gn_boxes final_box_nums gn_box_selected_indices \
    --output_onnx_file_path PDPostProcessing_reg_class_1x3x${H}x${W}_split_box.onnx
    onnxsim PDPostProcessing_reg_class_1x3x${H}x${W}_split_box.onnx PDPostProcessing_reg_class_1x3x${H}x${W}_split_box.onnx
    onnxsim PDPostProcessing_reg_class_1x3x${H}x${W}_split_box.onnx PDPostProcessing_reg_class_1x3x${H}x${W}_split_box.onnx
    sor4onnx \
    --input_onnx_file_path PDPostProcessing_reg_class_1x3x${H}x${W}_split_box.onnx \
    --old_new "PartitionedCall" "nms_box_gathernd" \
    --output_onnx_file_path PDPostProcessing_reg_class_1x3x${H}x${W}_split_box.onnx

    snc4onnx \
    --input_onnx_file_paths PDPostProcessing_reg_class_Nx3x${H}x${W}_split.onnx nms_box_gather_nd_N_${BOXES}.onnx \
    --srcop_destop cxcyw_wristcenterxy_middlefingerxy gn_boxes final_box_nums gn_box_selected_indices \
    --output_onnx_file_path PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box.onnx
    onnxsim PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box.onnx PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box.onnx
    onnxsim PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box.onnx PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box.onnx
    sor4onnx \
    --input_onnx_file_path PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box.onnx \
    --old_new "PartitionedCall" "nms_box_gathernd" \
    --output_onnx_file_path PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box.onnx

    ################################################### NonMaxSuppression + Score GatherND
    snc4onnx \
    --input_onnx_file_paths PDPostProcessing_reg_class_1x3x${H}x${W}_split_box.onnx nms_score_gather_nd_${BOXES}.onnx \
    --srcop_destop scores gn_scores workaround_mul_out gn_selected_indices \
    --output_onnx_file_path PDPostProcessing_reg_class_1x3x${H}x${W}_split_box_score.onnx
    onnxsim PDPostProcessing_reg_class_1x3x${H}x${W}_split_box_score.onnx PDPostProcessing_reg_class_1x3x${H}x${W}_split_box_score.onnx
    onnxsim PDPostProcessing_reg_class_1x3x${H}x${W}_split_box_score.onnx PDPostProcessing_reg_class_1x3x${H}x${W}_split_box_score.onnx
    sor4onnx \
    --input_onnx_file_path PDPostProcessing_reg_class_1x3x${H}x${W}_split_box_score.onnx \
    --old_new "model/tf.compat.v1.gather_nd/GatherNd" "nms_score_gathernd" \
    --output_onnx_file_path PDPostProcessing_reg_class_1x3x${H}x${W}_split_box_score.onnx
    sor4onnx \
    --input_onnx_file_path PDPostProcessing_reg_class_1x3x${H}x${W}_split_box_score.onnx \
    --old_new "PartitionedCall" "nms_score_gathernd_cast" \
    --output_onnx_file_path PDPostProcessing_reg_class_1x3x${H}x${W}_split_box_score.onnx

    snc4onnx \
    --input_onnx_file_paths PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box.onnx nms_score_gather_nd_N_${BOXES}.onnx \
    --srcop_destop scores gn_scores workaround_mul_out gn_selected_indices \
    --output_onnx_file_path PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box_score.onnx
    onnxsim PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box_score.onnx PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box_score.onnx
    onnxsim PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box_score.onnx PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box_score.onnx
    sor4onnx \
    --input_onnx_file_path PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box_score.onnx \
    --old_new "model/tf.compat.v1.gather_nd/GatherNd" "nms_score_gathernd" \
    --output_onnx_file_path PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box_score.onnx
    sor4onnx \
    --input_onnx_file_path PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box_score.onnx \
    --old_new "PartitionedCall" "nms_score_gathernd_cast" \
    --output_onnx_file_path PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box_score.onnx


    ################################################### NMS outputs merge
    python make_nms_outputs_merge.py
    onnxsim nms_scores_boxes_cat.onnx nms_scores_boxes_cat.onnx

    ################################################### merge
    snc4onnx \
    --input_onnx_file_paths PDPostProcessing_reg_class_1x3x${H}x${W}_split_box_score.onnx nms_scores_boxes_cat.onnx \
    --srcop_destop final_scores cat_score final_boxes cat_boxes \
    --output_onnx_file_path PDPostProcessing_reg_class_1x3x${H}x${W}_split_box_score_cat.onnx
    sod4onnx \
    --input_onnx_file_path PDPostProcessing_reg_class_1x3x${H}x${W}_split_box_score_cat.onnx \
    --output_op_names "final_batch_nums" \
    --output_onnx_file_path PDPostProcessing_reg_class_1x3x${H}x${W}_split_box_score_cat.onnx

    snc4onnx \
    --input_onnx_file_paths PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box_score.onnx nms_scores_boxes_cat.onnx \
    --srcop_destop final_scores cat_score final_boxes cat_boxes \
    --output_onnx_file_path PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box_score_cat.onnx
    sor4onnx \
    --input_onnx_file_path PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box_score_cat.onnx \
    --old_new "final_batch_nums" "batch_nums" \
    --output_onnx_file_path PDPostProcessing_reg_class_Nx3x${H}x${W}_split_box_score_cat.onnx
done

sam4onnx \
--input_onnx_file_path PDPostProcessing_reg_class_Nx3x128x128_split_box_score_cat.onnx \
--output_onnx_file_path PDPostProcessing_reg_class_Nx3x128x128_split_box_score_cat.onnx \
--op_name nonmaxsuppression11 \
--input_constants max_output_boxes_per_class int64 [1]
sam4onnx \
--input_onnx_file_path PDPostProcessing_reg_class_Nx3x128x128_split_box_score_cat.onnx \
--output_onnx_file_path PDPostProcessing_reg_class_Nx3x128x128_split_box_score_cat.onnx \
--op_name nonmaxsuppression11 \
--input_constants score_threshold float32 [\'-Infinity\']

sam4onnx \
--input_onnx_file_path PDPostProcessing_reg_class_Nx3x192x192_split_box_score_cat.onnx \
--output_onnx_file_path PDPostProcessing_reg_class_Nx3x192x192_split_box_score_cat.onnx \
--op_name nonmaxsuppression11 \
--input_constants max_output_boxes_per_class int64 [1]
sam4onnx \
--input_onnx_file_path PDPostProcessing_reg_class_Nx3x192x192_split_box_score_cat.onnx \
--output_onnx_file_path PDPostProcessing_reg_class_Nx3x192x192_split_box_score_cat.onnx \
--op_name nonmaxsuppression11 \
--input_constants score_threshold float32 [\'-Infinity\']
