#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1EoMp93492UEncY2tQMe0wrKqsVKBukCj" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1EoMp93492UEncY2tQMe0wrKqsVKBukCj" -o deeplabv3-mobilenetv3-small-cityscapes-5000-quant.tar.gz
tar -zxvf deeplabv3-mobilenetv3-small-cityscapes-5000-quant.tar.gz
rm deeplabv3-mobilenetv3-small-cityscapes-5000-quant.tar.gz

echo Download finished.
