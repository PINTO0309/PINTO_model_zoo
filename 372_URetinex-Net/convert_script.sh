#!/bin/bash

# t=0
#    1. model_Decom_low(input_low_img) -> P, Q
#       2. self.model_R(r=P, l=Q) -> R
#       3. self.model_L(l=Q) -> L
#
# t=1
#   4. self.P(I=input_low_img, Q=Q, R=R) -> P
#       5. self.Q(I=input_low_img, P=P, L=L) -> Q
#       6. self.model_R(r=P, l=Q) -> R
#       7. self.model_L(l=Q) -> L
#
# t=2
#   8. self.P(I=input_low_img, Q=Q, R=R) -> P
#       9. self.Q(I=input_low_img, P=P, L=L) -> Q
#       10. self.model_R(r=P, l=Q) -> R
#       11. self.model_L(l=Q) -> L
#
# 12. self.adjust_model(l=L, ratio=self.opts.ratio, R=R) -> final

RESOLUTIONS=(
    "180 320"
    "180 416"
    "180 512"
    "180 640"
    "180 800"
    "240 320"
    "240 416"
    "240 512"
    "240 640"
    "240 800"
    "240 960"
    "288 480"
    "288 512"
    "288 640"
    "288 800"
    "288 960"
    "288 1280"
    "320 320"
    "360 480"
    "360 512"
    "360 640"
    "360 800"
    "360 960"
    "360 1280"
    "416 416"
    "480 640"
    "480 800"
    "480 960"
    "480 1280"
    "512 512"
    "540 800"
    "540 960"
    "540 1280"
    "640 640"
    "720 1280"
)

for((j=0; j<${#RESOLUTIONS[@]}; j++))
do
    RESOLUTION=(`echo ${RESOLUTIONS[j]}`)
    H=${RESOLUTION[0]}
    W=${RESOLUTION[1]}
    echo @@@@@@@@@@@@@@@@@ processing ${H}x${W} ...

    #### 1 + 2 -> A
    snc4onnx \
    --input_onnx_file_paths uretinex_net_decom_low_${H}x${W}.onnx uretinex_net_r_${H}x${W}.onnx \
    --output_onnx_file_path 1_plus_2.onnx \
    --srcop_destop P input_P Q input_Q \
    --op_prefixes_after_merging pre1 pre2

    #### A + 3 -> B
    snc4onnx \
    --input_onnx_file_paths 1_plus_2.onnx uretinex_net_l_${H}x${W}.onnx \
    --output_onnx_file_path A_plus_3.onnx \
    --srcop_destop pre1_Q input_L \
    --op_prefixes_after_merging pre3 pre4

    ##############################################################################
    #### B + 4 -> C
    snc4onnx \
    --input_onnx_file_paths A_plus_3.onnx uretinex_net_p_t1_${H}x${W}.onnx \
    --output_onnx_file_path B_plus_4.onnx \
    --srcop_destop input input_P_t1 pre3_pre1_Q input_PQ_t1 R input_PR_t1 \
    --op_prefixes_after_merging pre5 pre6

    #### C + 5 -> D
    snc4onnx \
    --input_onnx_file_paths B_plus_4.onnx uretinex_net_q_t1_${H}x${W}.onnx \
    --output_onnx_file_path C_plus_5.onnx \
    --srcop_destop input input_Q_t1 P_t1 input_QP_t1 L input_QL_t1 \
    --op_prefixes_after_merging pre7 pre8

    #### D + 6 -> E
    snc4onnx \
    --input_onnx_file_paths C_plus_5.onnx uretinex_net_r_${H}x${W}.onnx \
    --output_onnx_file_path D_plus_6.onnx \
    --srcop_destop pre7_P_t1 input_P Q_t1 input_Q \
    --op_prefixes_after_merging pre9 pre10

    #### E + 7 -> F
    snc4onnx \
    --input_onnx_file_paths D_plus_6.onnx uretinex_net_l_${H}x${W}.onnx \
    --output_onnx_file_path E_plus_7.onnx \
    --srcop_destop pre9_Q_t1 input_L \
    --op_prefixes_after_merging pre11 pre12

    ##############################################################################
    #### F + 8 -> G
    snc4onnx \
    --input_onnx_file_paths E_plus_7.onnx uretinex_net_p_t2_${H}x${W}.onnx \
    --output_onnx_file_path F_plus_8.onnx \
    --srcop_destop input input_P_t2 pre11_pre9_Q_t1 input_PQ_t2 R input_PR_t2 \
    --op_prefixes_after_merging pre13 pre14

    #### G + 9 -> H
    snc4onnx \
    --input_onnx_file_paths F_plus_8.onnx uretinex_net_q_t2_${H}x${W}.onnx \
    --output_onnx_file_path G_plus_9.onnx \
    --srcop_destop input input_Q_t2 pre13_pre11_pre9_pre7_P_t1 input_QP_t2 L input_QL_t2 \
    --op_prefixes_after_merging pre15 pre16

    #### H + 10 -> I
    snc4onnx \
    --input_onnx_file_paths G_plus_9.onnx uretinex_net_r_${H}x${W}.onnx \
    --output_onnx_file_path H_plus_10.onnx \
    --srcop_destop P_t2 input_P Q_t2 input_Q \
    --op_prefixes_after_merging pre17 pre18

    #### I + 11 -> J
    snc4onnx \
    --input_onnx_file_paths H_plus_10.onnx uretinex_net_l_${H}x${W}.onnx \
    --output_onnx_file_path I_plus_11.onnx \
    --srcop_destop pre17_Q_t2 input_L \
    --op_prefixes_after_merging pre19 pre20


    ##############################################################################
    #### J + adjust -> final
    snc4onnx \
    --input_onnx_file_paths I_plus_11.onnx uretinex_net_adjust_${H}x${W}.onnx \
    --output_onnx_file_path uretinex_net_${H}x${W}.onnx \
    --srcop_destop R input_adjust_R L input_adjust_L \
    --op_prefixes_after_merging pre21 pre22

    ##############################################################################
    sor4onnx \
    --input_onnx_file_path uretinex_net_${H}x${W}.onnx \
    --old_new "pre21_input" "input" \
    --mode inputs \
    --search_mode prefix_match \
    --output_onnx_file_path uretinex_net_${H}x${W}.onnx

    sor4onnx \
    --input_onnx_file_path uretinex_net_${H}x${W}.onnx \
    --old_new "pre22_exposure_ratio_3to5" "exposure_ratio_3to5" \
    --mode inputs \
    --search_mode prefix_match \
    --output_onnx_file_path uretinex_net_${H}x${W}.onnx

    rm 1_plus_2.onnx
    rm A_plus_3.onnx
    rm B_plus_4.onnx
    rm C_plus_5.onnx
    rm D_plus_6.onnx
    rm E_plus_7.onnx
    rm F_plus_8.onnx
    rm G_plus_9.onnx
    rm H_plus_10.onnx
    rm I_plus_11.onnx
done


