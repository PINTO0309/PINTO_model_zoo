#!/bin/bash

############################################################ 1x3xHxW 6 person
python -m tf2onnx.convert \
--opset 11 \
--saved-model . \
--inputs-as-nchw input:0 \
--output movenet_multipose_lightning_HxW.onnx

snd4onnx \
--remove_node_names StatefulPartitionedCall/Cast StatefulPartitionedCall/sub \
--input_onnx_file_path movenet_multipose_lightning_HxW.onnx \
--output_onnx_file_path movenet_multipose_lightning_HxW.onnx

onnx2json \
--input_onnx_file_path movenet_multipose_lightning_HxW.onnx \
--output_json_path movenet_multipose_lightning_HxW.json \
--json_indent 2

sed -i -e 's/"elemType": 6/"elemType": 1/g' movenet_multipose_lightning_HxW.json
sed -i -e 's/:0//g' movenet_multipose_lightning_HxW.json
sed -i -e 's/output_0/output/g' movenet_multipose_lightning_HxW.json

json2onnx \
--input_json_path movenet_multipose_lightning_HxW.json \
--output_onnx_file_path movenet_multipose_lightning_HxW.onnx

onnxsim movenet_multipose_lightning_HxW.onnx movenet_multipose_lightning_HxW.onnx
onnxsim movenet_multipose_lightning_HxW.onnx movenet_multipose_lightning_HxW.onnx
onnxsim movenet_multipose_lightning_HxW.onnx movenet_multipose_lightning_HxW.onnx

rm movenet_multipose_lightning_HxW.json


################################################################ 2,5,10 person
PERSONS=2
python make_post-process.py \
--opset 11 \
--batches 1 \
--num_person ${PERSONS}

PERSONS=5
python make_post-process.py \
--opset 11 \
--batches 1 \
--num_person ${PERSONS}

PERSONS=10
python make_post-process.py \
--opset 11 \
--batches 1 \
--num_person ${PERSONS}

################################################################ 10 person
PERSONS=10
H=512
W=896
python -m tf2onnx.convert \
--opset 11 \
--saved-model . \
--inputs-as-nchw input:0 \
--inputs input:0[1,${H},${W},3] \
--output movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/Cast StatefulPartitionedCall/sub \
--input_onnx_file_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx \
--output_onnx_file_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx
onnx2json \
--input_onnx_file_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx \
--output_json_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json \
--json_indent 2
sed -i -e 's/"elemType": 6/"elemType": 1/g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
sed -i -e 's/:0//g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
sed -i -e 's/output_0/output/g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [${PERSONS}] --dtype int64 --mode encode`
sed -i -e 's#"BgAAAAAAAAA="#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [1,${PERSONS},2] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAAGAAAAAAAAAAIAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [${PERSONS},1] --dtype int64 --mode encode`
sed -i -e 's#"BgAAAAAAAAABAAAAAAAAAA=="#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [${PERSONS},17,2] --dtype int64 --mode encode`
sed -i -e 's#"BgAAAAAAAAARAAAAAAAAAAIAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [1,1,${PERSONS}] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAABAAAAAAAAAAYAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [1,${PERSONS},17] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAAGAAAAAAAAABEAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [1,${PERSONS},17,2] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAAGAAAAAAAAABEAAAAAAAAAAgAAAAAAAAA="#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [1,${PERSONS},-1] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAAGAAAAAAAAAP//////////"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
keypoints_number=$((17 * PERSONS))
base64str=`sed4onnx --constant_string [-1,${keypoints_number}] --dtype int64 --mode encode`
sed -i -e 's#"//////////9mAAAAAAAAAA=="#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
############################################################################### StatefulPartitionedCall/zeros_like
sed -i -z -e 's/"1",\n          "6",\n          "17"/"1",\n          "'${PERSONS}'",\n          "17"/g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json   # StatefulPartitionedCall/zeros_like
zeros=()
for (( i=0; i<$keypoints_number; i++ )); do
    zeros+=("0")
done
zeroslist=$(IFS=,; echo "[${zeros[*]}]")
base64str=`sed4onnx --constant_string ${zeroslist} --dtype int32 --mode encode`
sed -i -e 's#"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
############################################################################### StatefulPartitionedCall/concat_1
sed -i -e 's#"dimValue": "6"#"dimValue": "'${PERSONS}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
############################################################################### StatefulPartitionedCall/stack_6_Concat__xxx - const_fold_opt__xxx
sed -i -z -e 's#"dims": \[\n          "102",\n          "1"\n        \],#"dims": \[\n          "'${keypoints_number}'",\n          "1"\n        \],#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
values=()
for (( j=0; j<$PERSONS; j++ )); do
    for (( i=0; i<=16; i++ )); do
        values+=("$i")
    done
done
seventeenlist=$(IFS=,; echo "[${values[*]}]")
base64str=`sed4onnx --constant_string ${seventeenlist} --dtype int32 --mode encode`
sed -i -e 's#"AAAAAAEAAAACAAAAAwAAAAQAAAAFAAAABgAAAAcAAAAIAAAACQAAAAoAAAALAAAADAAAAA0AAAAOAAAADwAAABAAAAAAAAAAAQAAAAIAAAADAAAABAAAAAUAAAAGAAAABwAAAAgAAAAJAAAACgAAAAsAAAAMAAAADQAAAA4AAAAPAAAAEAAAAAAAAAABAAAAAgAAAAMAAAAEAAAABQAAAAYAAAAHAAAACAAAAAkAAAAKAAAACwAAAAwAAAANAAAADgAAAA8AAAAQAAAAAAAAAAEAAAACAAAAAwAAAAQAAAAFAAAABgAAAAcAAAAIAAAACQAAAAoAAAALAAAADAAAAA0AAAAOAAAADwAAABAAAAAAAAAAAQAAAAIAAAADAAAABAAAAAUAAAAGAAAABwAAAAgAAAAJAAAACgAAAAsAAAAMAAAADQAAAA4AAAAPAAAAEAAAAAAAAAABAAAAAgAAAAMAAAAEAAAABQAAAAYAAAAHAAAACAAAAAkAAAAKAAAACwAAAAwAAAANAAAADgAAAA8AAAAQAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json

###############################################################################
###############################################################################
###############################################################################
zeros=()
for (( i=0; i<$PERSONS; i++ )); do
    zeros+=("0")
done
zeroslist=$(IFS=,; echo "[${zeros[*]}]")
base64str=`sed4onnx --constant_string ${zeroslist} --dtype int32 --mode encode`
concat_const_number=710
sed -i -z -e 's#      {\n        "dims": \[\n          "6",\n          "1"\n        \],\n        "dataType": 6,\n        "name": "const_fold_opt__'${concat_const_number}'",\n        "rawData": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"\n      },#      {\n        "dims": \[\n          "'${PERSONS}'",\n          "1"\n        \],\n        "dataType": 6,\n        "name": "const_fold_opt__'${concat_const_number}'",\n        "rawData": "'${base64str}'"\n      },#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
###############################################################################
###############################################################################
###############################################################################

############################################################################### Generate ONNX
json2onnx \
--input_json_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json \
--output_onnx_file_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx

onnxsim movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx
onnxsim movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx
onnxsim movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx

sit4onnx -if movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx -oep cpu

rm movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json




################################################################ 5 person
PERSONS=5
H=512
W=896
python -m tf2onnx.convert \
--opset 11 \
--saved-model . \
--inputs-as-nchw input:0 \
--inputs input:0[1,${H},${W},3] \
--output movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/Cast StatefulPartitionedCall/sub \
--input_onnx_file_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx \
--output_onnx_file_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx
onnx2json \
--input_onnx_file_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx \
--output_json_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json \
--json_indent 2
sed -i -e 's/"elemType": 6/"elemType": 1/g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
sed -i -e 's/:0//g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
sed -i -e 's/output_0/output/g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
# BgAAAAAAAAA=: int64 [6]
# BQAAAAAAAAA=: int64 [5]
base64str=`sed4onnx --constant_string [${PERSONS}] --dtype int64 --mode encode`
sed -i -e 's#"BgAAAAAAAAA="#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
# AQAAAAAAAAAGAAAAAAAAAAIAAAAAAAAA: int64 [1,6,2]
# AQAAAAAAAAAFAAAAAAAAAAIAAAAAAAAA: int64 [1,5,2]
base64str=`sed4onnx --constant_string [1,${PERSONS},2] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAAGAAAAAAAAAAIAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
# BgAAAAAAAAABAAAAAAAAAA==: int64 [6,1]
# BQAAAAAAAAABAAAAAAAAAA==: int64 [5,1]
base64str=`sed4onnx --constant_string [${PERSONS},1] --dtype int64 --mode encode`
sed -i -e 's#"BgAAAAAAAAABAAAAAAAAAA=="#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
# BgAAAAAAAAARAAAAAAAAAAIAAAAAAAAA: int64 [6,17,2]
# BQAAAAAAAAARAAAAAAAAAAIAAAAAAAAA: int64 [5,17,2]
base64str=`sed4onnx --constant_string [${PERSONS},17,2] --dtype int64 --mode encode`
sed -i -e 's#"BgAAAAAAAAARAAAAAAAAAAIAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
# AQAAAAAAAAABAAAAAAAAAAYAAAAAAAAA: [1,1,6]
# AQAAAAAAAAABAAAAAAAAAAUAAAAAAAAA: [1,1,5]
base64str=`sed4onnx --constant_string [1,1,${PERSONS}] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAABAAAAAAAAAAYAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
# AQAAAAAAAAAGAAAAAAAAABEAAAAAAAAA: int64 [1,6,17]
# AQAAAAAAAAAFAAAAAAAAABEAAAAAAAAA: int64 [1,5,17]
base64str=`sed4onnx --constant_string [1,${PERSONS},17] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAAGAAAAAAAAABEAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
# AQAAAAAAAAAGAAAAAAAAABEAAAAAAAAAAgAAAAAAAAA=: [1,6,17,2]
# AQAAAAAAAAAFAAAAAAAAABEAAAAAAAAAAgAAAAAAAAA=: [1,5,17,2]
base64str=`sed4onnx --constant_string [1,${PERSONS},17,2] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAAGAAAAAAAAABEAAAAAAAAAAgAAAAAAAAA="#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
# AQAAAAAAAAAGAAAAAAAAAP//////////: int64 [1,6,-1]
# AQAAAAAAAAAFAAAAAAAAAP//////////: int64 [1,5,-1]
base64str=`sed4onnx --constant_string [1,${PERSONS},-1] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAAGAAAAAAAAAP//////////"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
# noqa: F401　//////////9mAAAAAAAAAA==: int64 [-1,102]
# noqa: F401　//////////9VAAAAAAAAAA==: int64 [-1,85]
keypoints_number=$((17 * PERSONS))
base64str=`sed4onnx --constant_string [-1,${keypoints_number}] --dtype int64 --mode encode`
sed -i -e 's#"//////////9mAAAAAAAAAA=="#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
############################################################################### StatefulPartitionedCall/zeros_like
sed -i -z -e 's/"1",\n          "6",\n          "17"/"1",\n          "'${PERSONS}'",\n          "17"/g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json   # StatefulPartitionedCall/zeros_like
# 102: int32 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# 85 : int32 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
zeros=()
for (( i=0; i<$keypoints_number; i++ )); do
    zeros+=("0")
done
zeroslist=$(IFS=,; echo "[${zeros[*]}]")
base64str=`sed4onnx --constant_string ${zeroslist} --dtype int32 --mode encode`
sed -i -e 's#"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
############################################################################### StatefulPartitionedCall/concat_1
sed -i -e 's#"dimValue": "6"#"dimValue": "'${PERSONS}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
############################################################################### StatefulPartitionedCall/stack_6_Concat__xxx - const_fold_opt__xxx
sed -i -z -e 's#"dims": \[\n          "102",\n          "1"\n        \],#"dims": \[\n          "'${keypoints_number}'",\n          "1"\n        \],#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
# 6: int32 [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# 5: int32 [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
values=()
for (( j=0; j<$PERSONS; j++ )); do
    for (( i=0; i<=16; i++ )); do
        values+=("$i")
    done
done
seventeenlist=$(IFS=,; echo "[${values[*]}]")
base64str=`sed4onnx --constant_string ${seventeenlist} --dtype int32 --mode encode`
sed -i -e 's#"AAAAAAEAAAACAAAAAwAAAAQAAAAFAAAABgAAAAcAAAAIAAAACQAAAAoAAAALAAAADAAAAA0AAAAOAAAADwAAABAAAAAAAAAAAQAAAAIAAAADAAAABAAAAAUAAAAGAAAABwAAAAgAAAAJAAAACgAAAAsAAAAMAAAADQAAAA4AAAAPAAAAEAAAAAAAAAABAAAAAgAAAAMAAAAEAAAABQAAAAYAAAAHAAAACAAAAAkAAAAKAAAACwAAAAwAAAANAAAADgAAAA8AAAAQAAAAAAAAAAEAAAACAAAAAwAAAAQAAAAFAAAABgAAAAcAAAAIAAAACQAAAAoAAAALAAAADAAAAA0AAAAOAAAADwAAABAAAAAAAAAAAQAAAAIAAAADAAAABAAAAAUAAAAGAAAABwAAAAgAAAAJAAAACgAAAAsAAAAMAAAADQAAAA4AAAAPAAAAEAAAAAAAAAABAAAAAgAAAAMAAAAEAAAABQAAAAYAAAAHAAAACAAAAAkAAAAKAAAACwAAAAwAAAANAAAADgAAAA8AAAAQAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
###############################################################################
###############################################################################
###############################################################################
# 6: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA: int32 [0,0,0,0,0,0]
# 5: AAAAAAAAAAAAAAAAAAAAAAAAAAA=: int32 [0,0,0,0,0]
zeros=()
for (( i=0; i<$PERSONS; i++ )); do
    zeros+=("0")
done
zeroslist=$(IFS=,; echo "[${zeros[*]}]")
base64str=`sed4onnx --constant_string ${zeroslist} --dtype int32 --mode encode`
concat_const_number=694
sed -i -z -e 's#      {\n        "dims": \[\n          "6",\n          "1"\n        \],\n        "dataType": 6,\n        "name": "const_fold_opt__'${concat_const_number}'",\n        "rawData": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"\n      },#      {\n        "dims": \[\n          "'${PERSONS}'",\n          "1"\n        \],\n        "dataType": 6,\n        "name": "const_fold_opt__'${concat_const_number}'",\n        "rawData": "'${base64str}'"\n      },#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json

############################################################################### Generate ONNX
json2onnx \
--input_json_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json \
--output_onnx_file_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx

onnxsim movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx
onnxsim movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx
onnxsim movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx

sit4onnx -if movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx -oep cpu

rm movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json



################################################################ 2 person
PERSONS=2
H=512
W=896
python -m tf2onnx.convert \
--opset 11 \
--saved-model . \
--inputs-as-nchw input:0 \
--inputs input:0[1,${H},${W},3] \
--output movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/Cast StatefulPartitionedCall/sub \
--input_onnx_file_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx \
--output_onnx_file_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx
onnx2json \
--input_onnx_file_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx \
--output_json_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json \
--json_indent 2
sed -i -e 's/"elemType": 6/"elemType": 1/g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
sed -i -e 's/:0//g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
sed -i -e 's/output_0/output/g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [${PERSONS}] --dtype int64 --mode encode`
sed -i -e 's#"BgAAAAAAAAA="#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [1,${PERSONS},2] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAAGAAAAAAAAAAIAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [${PERSONS},1] --dtype int64 --mode encode`
sed -i -e 's#"BgAAAAAAAAABAAAAAAAAAA=="#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [${PERSONS},17,2] --dtype int64 --mode encode`
sed -i -e 's#"BgAAAAAAAAARAAAAAAAAAAIAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [1,1,${PERSONS}] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAABAAAAAAAAAAYAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [1,${PERSONS},17] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAAGAAAAAAAAABEAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [1,${PERSONS},17,2] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAAGAAAAAAAAABEAAAAAAAAAAgAAAAAAAAA="#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
base64str=`sed4onnx --constant_string [1,${PERSONS},-1] --dtype int64 --mode encode`
sed -i -e 's#"AQAAAAAAAAAGAAAAAAAAAP//////////"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
keypoints_number=$((17 * PERSONS))
base64str=`sed4onnx --constant_string [-1,${keypoints_number}] --dtype int64 --mode encode`
sed -i -e 's#"//////////9mAAAAAAAAAA=="#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
############################################################################### StatefulPartitionedCall/zeros_like
sed -i -z -e 's/"1",\n          "6",\n          "17"/"1",\n          "'${PERSONS}'",\n          "17"/g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json   # StatefulPartitionedCall/zeros_like
zeros=()
for (( i=0; i<$keypoints_number; i++ )); do
    zeros+=("0")
done
zeroslist=$(IFS=,; echo "[${zeros[*]}]")
base64str=`sed4onnx --constant_string ${zeroslist} --dtype int32 --mode encode`
sed -i -e 's#"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
############################################################################### StatefulPartitionedCall/concat_1
sed -i -e 's#"dimValue": "6"#"dimValue": "'${PERSONS}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
############################################################################### StatefulPartitionedCall/stack_6_Concat__xxx - const_fold_opt__xxx
sed -i -z -e 's#"dims": \[\n          "102",\n          "1"\n        \],#"dims": \[\n          "'${keypoints_number}'",\n          "1"\n        \],#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
values=()
for (( j=0; j<$PERSONS; j++ )); do
    for (( i=0; i<=16; i++ )); do
        values+=("$i")
    done
done
seventeenlist=$(IFS=,; echo "[${values[*]}]")
base64str=`sed4onnx --constant_string ${seventeenlist} --dtype int32 --mode encode`
sed -i -e 's#"AAAAAAEAAAACAAAAAwAAAAQAAAAFAAAABgAAAAcAAAAIAAAACQAAAAoAAAALAAAADAAAAA0AAAAOAAAADwAAABAAAAAAAAAAAQAAAAIAAAADAAAABAAAAAUAAAAGAAAABwAAAAgAAAAJAAAACgAAAAsAAAAMAAAADQAAAA4AAAAPAAAAEAAAAAAAAAABAAAAAgAAAAMAAAAEAAAABQAAAAYAAAAHAAAACAAAAAkAAAAKAAAACwAAAAwAAAANAAAADgAAAA8AAAAQAAAAAAAAAAEAAAACAAAAAwAAAAQAAAAFAAAABgAAAAcAAAAIAAAACQAAAAoAAAALAAAADAAAAA0AAAAOAAAADwAAABAAAAAAAAAAAQAAAAIAAAADAAAABAAAAAUAAAAGAAAABwAAAAgAAAAJAAAACgAAAAsAAAAMAAAADQAAAA4AAAAPAAAAEAAAAAAAAAABAAAAAgAAAAMAAAAEAAAABQAAAAYAAAAHAAAACAAAAAkAAAAKAAAACwAAAAwAAAANAAAADgAAAA8AAAAQAAAA"#"'${base64str}'"#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json

###############################################################################
###############################################################################
###############################################################################
zeros=()
for (( i=0; i<$PERSONS; i++ )); do
    zeros+=("0")
done
zeroslist=$(IFS=,; echo "[${zeros[*]}]")
base64str=`sed4onnx --constant_string ${zeroslist} --dtype int32 --mode encode`
concat_const_number=710
sed -i -z -e 's#      {\n        "dims": \[\n          "6",\n          "1"\n        \],\n        "dataType": 6,\n        "name": "const_fold_opt__'${concat_const_number}'",\n        "rawData": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"\n      },#      {\n        "dims": \[\n          "'${PERSONS}'",\n          "1"\n        \],\n        "dataType": 6,\n        "name": "const_fold_opt__'${concat_const_number}'",\n        "rawData": "'${base64str}'"\n      },#g' movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
###############################################################################
###############################################################################
###############################################################################

############################################################################### Generate ONNX
json2onnx \
--input_json_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json \
--output_onnx_file_path movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx

onnxsim movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx
onnxsim movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx
onnxsim movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx

sit4onnx -if movenet_multipose_lightning_${H}x${W}_p${PERSONS}.onnx -oep cpu

rm movenet_multipose_lightning_${H}x${W}_p${PERSONS}.json
