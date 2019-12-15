#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1HBPDA8gMVqMhJMiOj1HT0t8ijSHeSYAK" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1HBPDA8gMVqMhJMiOj1HT0t8ijSHeSYAK" -o deeplabv3-mobilenetv3-large-cityscapes-4361-quant.tar.gz
tar -zxvf deeplabv3-mobilenetv3-large-cityscapes-4361-quant.tar.gz
rm deeplabv3-mobilenetv3-large-cityscapes-4361-quant.tar.gz

echo Download finished.
