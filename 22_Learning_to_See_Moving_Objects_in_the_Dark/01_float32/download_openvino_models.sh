#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=11sTiZn_v6pNMhLZ9xq_1iTXwilIf_Xwj" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=11sTiZn_v6pNMhLZ9xq_1iTXwilIf_Xwj" -o openvino_models.tar.gz
tar -zxvf openvino_models.tar.gz
rm openvino_models.tar.gz
echo Download finished.