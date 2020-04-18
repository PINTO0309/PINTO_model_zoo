#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1uk2dSu47CNtrX4Q5PiG9Wa_gzjKWLq5j" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1uk2dSu47CNtrX4Q5PiG9Wa_gzjKWLq5j" -o deeplabv3-mobilenetv3-small-voc-500000.tar.gz
tar -zxvf deeplabv3-mobilenetv3-small-voc-500000.tar.gz
rm deeplabv3-mobilenetv3-small-voc-500000.tar.gz

echo Download finished.
