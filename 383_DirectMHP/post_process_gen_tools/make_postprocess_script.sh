#!/bin/bash

pip install -U pip \
&& pip install onnxsim==0.4.17 \
&& pip install -U simple-onnx-processing-tools \
&& pip install onnx==1.13.1 \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

OPSET=11
BATCHES=1
CLASSES=1

MODELS=(
    "directmhp_300wlp_m_finetune"
    "directmhp_300wlp_s_finetune"
    "directmhp_agora_m"
    "directmhp_agora_s"
    "directmhp_cmu_m"
    "directmhp_cmu_s"
)

RESOLUTIONS=(
    "192 320 3825"
    "192 640 7650"
    "256 320 5100"
    "256 640 10200"
    "256 960 15300"
    "320 320 6375"
    "384 640 15300"
    "384 960 22950"
    "384 1280 30600"
    "512 512 16320"
    "512 640 20400"
    "512 960 30600"
    "512 1280 40800"
    "640 640 25500"
    "768 1280 61200"
)

for((i=0; i<${#MODELS[@]}; i++))
do
    MODEL=(`echo ${MODELS[i]}`)

    for((j=0; j<${#RESOLUTIONS[@]}; j++))
    do
        RESOLUTION=(`echo ${RESOLUTIONS[j]}`)
        H=${RESOLUTION[0]}
        W=${RESOLUTION[1]}
        BOXES=${RESOLUTION[2]}
        echo @@@@@@@@@@@@@@@@@ processing ${MODEL}_${H}x${W} ...

        ################################################### Boxes + Scores
        python make_boxes_scores.py -o ${OPSET} -b ${BATCHES} -x ${BOXES} -c ${CLASSES}
        python make_cxcywh_y1x1y2x2.py -o ${OPSET} -b ${BATCHES} -x ${BOXES}

        snc4onnx \
        --input_onnx_file_paths 01_boxes_scores_${BOXES}.onnx 02_cxcywh_x1y1x2y2_y1x1y2x2_${BOXES}.onnx \
        --srcop_destop boxes_cxcywh cxcywh \
        --op_prefixes_after_merging box cxcy \
        --output_onnx_file_path 03_boxes_x1y1x2y2_y1x1y2x2_scores_${BOXES}.onnx

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
        --attributes value float32 [0.5] \
        --output_onnx_file_path 05_Constant_iou_threshold.onnx

        sog4onnx \
        --op_type Constant \
        --opset ${OPSET} \
        --op_name score_threshold_const \
        --output_variables score_threshold float32 [1] \
        --attributes value float32 [0.25] \
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
        --input_onnx_file_paths 04_Constant_max_output_boxes_per_class.onnx 07_NonMaxSuppression${OPSET}.onnx \
        --srcop_destop max_output_boxes_per_class max_output_boxes_per_class_var \
        --output_onnx_file_path 07_NonMaxSuppression${OPSET}.onnx

        snc4onnx \
        --input_onnx_file_paths 05_Constant_iou_threshold.onnx 07_NonMaxSuppression${OPSET}.onnx \
        --srcop_destop iou_threshold iou_threshold_var \
        --output_onnx_file_path 07_NonMaxSuppression${OPSET}.onnx

        snc4onnx \
        --input_onnx_file_paths 06_Constant_score_threshold.onnx 07_NonMaxSuppression${OPSET}.onnx \
        --srcop_destop score_threshold score_threshold_var \
        --output_onnx_file_path 07_NonMaxSuppression${OPSET}.onnx

        soc4onnx \
        --input_onnx_file_path 07_NonMaxSuppression${OPSET}.onnx \
        --output_onnx_file_path 07_NonMaxSuppression${OPSET}.onnx \
        --opset ${OPSET}

        ################################################### Boxes + Scores + NonMaxSuppression
        snc4onnx \
        --input_onnx_file_paths 03_boxes_x1y1x2y2_y1x1y2x2_scores_${BOXES}.onnx 07_NonMaxSuppression${OPSET}.onnx \
        --srcop_destop scores scores_var y1x1y2x2 boxes_var \
        --output_onnx_file_path 08_nms_directmhp_${BOXES}.onnx

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
        --input_onnx_file_paths 10_Constant_workaround_mul.onnx 09_Mul${OPSET}_workaround.onnx \
        --srcop_destop workaround_mul_const workaround_mul_b \
        --output_onnx_file_path 11_Mul${OPSET}_workaround.onnx

        ################################################### NonMaxSuppression + Myriad workaround Mul
        snc4onnx \
        --input_onnx_file_paths 08_nms_directmhp_${BOXES}.onnx 11_Mul${OPSET}_workaround.onnx \
        --srcop_destop selected_indices workaround_mul_a \
        --output_onnx_file_path 12_nms_directmhp_${BOXES}.onnx

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
        --search_mode suffix_match \
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

        ################################################### NonMaxSuppression + Score GatherND
        snc4onnx \
        --input_onnx_file_paths 12_nms_directmhp_${BOXES}.onnx 13_nms_score_gather_nd.onnx \
        --srcop_destop scores gn_scores workaround_mul_out gn_selected_indices \
        --output_onnx_file_path 14_nms_directmhp_${BOXES}_nd.onnx

        onnxsim 14_nms_directmhp_${BOXES}_nd.onnx 14_nms_directmhp_${BOXES}_nd.onnx
        onnxsim 14_nms_directmhp_${BOXES}_nd.onnx 14_nms_directmhp_${BOXES}_nd.onnx

        ################################################### Final Batch Nums
        python make_final_batch_nums_final_class_nums_final_box_nums.py

        ################################################### Boxes GatherND
        python make_box_gather_nd.py -b ${BATCHES} -x ${BOXES}

        python -m tf2onnx.convert \
        --opset ${OPSET} \
        --tflite saved_model_postprocess/nms_box_gather_nd.tflite \
        --output 16_nms_box_gather_nd.onnx

        sor4onnx \
        --input_onnx_file_path 16_nms_box_gather_nd.onnx \
        --old_new ":0" "" \
        --mode full \
        --search_mode suffix_match \
        --output_onnx_file_path 16_nms_box_gather_nd.onnx

        sor4onnx \
        --input_onnx_file_path 16_nms_box_gather_nd.onnx \
        --old_new "serving_default_input_1" "gn_boxes" \
        --output_onnx_file_path 16_nms_box_gather_nd.onnx \
        --mode inputs

        sor4onnx \
        --input_onnx_file_path 16_nms_box_gather_nd.onnx \
        --old_new "serving_default_input_2" "gn_box_selected_indices" \
        --output_onnx_file_path 16_nms_box_gather_nd.onnx \
        --mode inputs

        sor4onnx \
        --input_onnx_file_path 16_nms_box_gather_nd.onnx \
        --old_new "PartitionedCall" "final_boxes" \
        --output_onnx_file_path 16_nms_box_gather_nd.onnx \
        --mode outputs

        python make_input_output_shape_update.py \
        --input_onnx_file_path 16_nms_box_gather_nd.onnx \
        --output_onnx_file_path 16_nms_box_gather_nd.onnx \
        --input_names gn_boxes \
        --input_names gn_box_selected_indices \
        --input_shapes ${BATCHES} ${BOXES} 4 \
        --input_shapes N 2 \
        --output_names final_boxes \
        --output_shapes N 4

        onnxsim 16_nms_box_gather_nd.onnx 16_nms_box_gather_nd.onnx
        onnxsim 16_nms_box_gather_nd.onnx 16_nms_box_gather_nd.onnx

        ################################################### nms_directmhp_xxx_nd + nms_final_batch_nums_final_class_nums_final_box_nums
        snc4onnx \
        --input_onnx_file_paths 14_nms_directmhp_${BOXES}_nd.onnx 15_nms_final_batch_nums_final_class_nums_final_box_nums.onnx \
        --srcop_destop workaround_mul_out bc_input \
        --op_prefixes_after_merging main01 sub01 \
        --output_onnx_file_path 17_nms_directmhp_${BOXES}_split.onnx

        sor4onnx \
        --input_onnx_file_path 17_nms_directmhp_${BOXES}_split.onnx \
        --old_new "main01_nonmaxsuppression11" "NonMaxSuppression11" \
        --output_onnx_file_path 18_nms_directmhp_${BOXES}_split.onnx \
        --mode full

        sor4onnx \
        --input_onnx_file_path 18_nms_directmhp_${BOXES}_split.onnx \
        --old_new "main01_y1x1y2x2" "y1x1y2x2" \
        --output_onnx_file_path 18_nms_directmhp_${BOXES}_split.onnx \
        --mode full

        sor4onnx \
        --input_onnx_file_path 18_nms_directmhp_${BOXES}_split.onnx \
        --old_new "main01_scores" "scores" \
        --output_onnx_file_path 18_nms_directmhp_${BOXES}_split.onnx \
        --mode full

        sor4onnx \
        --input_onnx_file_path 18_nms_directmhp_${BOXES}_split.onnx \
        --old_new "main01_max_output_boxes_per_class" "max_output_boxes_per_class" \
        --output_onnx_file_path 18_nms_directmhp_${BOXES}_split.onnx \
        --mode full

        sor4onnx \
        --input_onnx_file_path 18_nms_directmhp_${BOXES}_split.onnx \
        --old_new "main01_iou_threshold" "iou_threshold" \
        --output_onnx_file_path 18_nms_directmhp_${BOXES}_split.onnx \
        --mode full

        sor4onnx \
        --input_onnx_file_path 18_nms_directmhp_${BOXES}_split.onnx \
        --old_new "main01_score_threshold" "score_threshold" \
        --output_onnx_file_path 18_nms_directmhp_${BOXES}_split.onnx \
        --mode full

        sor4onnx \
        --input_onnx_file_path 18_nms_directmhp_${BOXES}_split.onnx \
        --old_new "main01_selected_indices" "selected_indices" \
        --output_onnx_file_path 18_nms_directmhp_${BOXES}_split.onnx \
        --mode full

        ################################################### nms_directmhp_${BOXES}_split + nms_box_gather_nd
        snc4onnx \
        --input_onnx_file_paths 18_nms_directmhp_${BOXES}_split.onnx 16_nms_box_gather_nd.onnx \
        --srcop_destop x1y1x2y2 gn_boxes final_box_nums gn_box_selected_indices \
        --output_onnx_file_path 19_nms_directmhp_${BOXES}_merged.onnx

        onnxsim 19_nms_directmhp_${BOXES}_merged.onnx 19_nms_directmhp_${BOXES}_merged.onnx
        onnxsim 19_nms_directmhp_${BOXES}_merged.onnx 19_nms_directmhp_${BOXES}_merged.onnx

        ################################################### PitchYawRoll GatherND
        python make_pitchyawroll_gather_nd.py -b ${BATCHES} -x ${BOXES}

        python -m tf2onnx.convert \
        --opset ${OPSET} \
        --tflite saved_model_postprocess/nms_pitchyawroll_gather_nd.tflite \
        --output 20_nms_pitchyawroll_gather_nd.onnx

        sor4onnx \
        --input_onnx_file_path 20_nms_pitchyawroll_gather_nd.onnx \
        --old_new ":0" "" \
        --mode full \
        --search_mode suffix_match \
        --output_onnx_file_path 20_nms_pitchyawroll_gather_nd.onnx

        sor4onnx \
        --input_onnx_file_path 20_nms_pitchyawroll_gather_nd.onnx \
        --old_new "serving_default_input_1" "gn_pitchyawroll" \
        --output_onnx_file_path 20_nms_pitchyawroll_gather_nd.onnx \
        --mode inputs

        sor4onnx \
        --input_onnx_file_path 20_nms_pitchyawroll_gather_nd.onnx \
        --old_new "serving_default_input_2" "gn_pitchyawroll_selected_indices" \
        --output_onnx_file_path 20_nms_pitchyawroll_gather_nd.onnx \
        --mode inputs

        sor4onnx \
        --input_onnx_file_path 20_nms_pitchyawroll_gather_nd.onnx \
        --old_new "PartitionedCall" "final_pitchyawroll" \
        --output_onnx_file_path 20_nms_pitchyawroll_gather_nd.onnx \
        --mode outputs

        python make_input_output_shape_update.py \
        --input_onnx_file_path 20_nms_pitchyawroll_gather_nd.onnx \
        --output_onnx_file_path 20_nms_pitchyawroll_gather_nd.onnx \
        --input_names gn_pitchyawroll \
        --input_names gn_pitchyawroll_selected_indices \
        --input_shapes ${BATCHES} ${BOXES} 3 \
        --input_shapes N 2 \
        --output_names final_pitchyawroll \
        --output_shapes N 3

        onnxsim 20_nms_pitchyawroll_gather_nd.onnx 20_nms_pitchyawroll_gather_nd.onnx
        onnxsim 20_nms_pitchyawroll_gather_nd.onnx 20_nms_pitchyawroll_gather_nd.onnx

        sor4onnx \
        --input_onnx_file_path 19_nms_directmhp_${BOXES}_merged.onnx \
        --old_new "main01_model/tf.compat.v1.gather_nd/GatherNd" "scores_nd" \
        --output_onnx_file_path 19_nms_directmhp_${BOXES}_merged.onnx \
        --mode full

        sor4onnx \
        --input_onnx_file_path 19_nms_directmhp_${BOXES}_merged.onnx \
        --old_new "PartitionedCall" "boxes_nd" \
        --output_onnx_file_path 19_nms_directmhp_${BOXES}_merged.onnx \
        --mode full

        sor4onnx \
        --input_onnx_file_path 20_nms_pitchyawroll_gather_nd.onnx \
        --old_new "PartitionedCall" "pitchyawroll_nd" \
        --output_onnx_file_path 20_nms_pitchyawroll_gather_nd.onnx \
        --mode full

        ################################################### nms_directmhp_${BOXES}_split + nms_pitchyawroll_gather_nd
        snc4onnx \
        --input_onnx_file_paths 19_nms_directmhp_${BOXES}_merged.onnx 20_nms_pitchyawroll_gather_nd.onnx \
        --srcop_destop pitch_yaw_roll gn_pitchyawroll final_box_nums gn_pitchyawroll_selected_indices \
        --output_onnx_file_path 21_nms_directmhp_${BOXES}_merged.onnx

        onnxsim 21_nms_directmhp_${BOXES}_merged.onnx 21_nms_directmhp_${BOXES}_merged.onnx
        onnxsim 21_nms_directmhp_${BOXES}_merged.onnx 21_nms_directmhp_${BOXES}_merged.onnx

        ################################################### nms output merge
        python make_nms_outputs_merge.py

        onnxsim 22_nms_batchno_classid_x1y1x2y2_score_pitchyawroll_cat.onnx 22_nms_batchno_classid_x1y1x2y2_score_pitchyawroll_cat.onnx

        ################################################### merge
        snc4onnx \
        --input_onnx_file_paths 21_nms_directmhp_${BOXES}_merged.onnx 22_nms_batchno_classid_x1y1x2y2_score_pitchyawroll_cat.onnx \
        --srcop_destop final_batch_nums cat_batch final_class_nums cat_classid final_boxes cat_x1y1x2y2 final_scores cat_score final_pitchyawroll cat_pitchyawroll \
        --output_onnx_file_path 23_nms_directmhp_${BOXES}.onnx

        ################################################### directmhp + Post-Process
        snc4onnx \
        --input_onnx_file_paths nopost/${MODEL}_${H}x${W}.onnx 23_nms_directmhp_${BOXES}.onnx \
        --srcop_destop output predictions \
        --output_onnx_file_path withpost/${MODEL}_post_${H}x${W}.onnx

        onnxsim withpost/${MODEL}_post_${H}x${W}.onnx withpost/${MODEL}_post_${H}x${W}.onnx
        onnxsim withpost/${MODEL}_post_${H}x${W}.onnx withpost/${MODEL}_post_${H}x${W}.onnx
    done
done
