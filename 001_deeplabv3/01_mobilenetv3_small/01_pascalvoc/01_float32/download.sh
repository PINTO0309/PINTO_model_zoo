#!/bin/bash

fileid="1uk2dSu47CNtrX4Q5PiG9Wa_gzjKWLq5j"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o deeplabv3-mobilenetv3-small-voc-500000.tar.gz
tar -zxvf deeplabv3-mobilenetv3-small-voc-500000.tar.gz
rm deeplabv3-mobilenetv3-small-voc-500000.tar.gz

echo Download finished.