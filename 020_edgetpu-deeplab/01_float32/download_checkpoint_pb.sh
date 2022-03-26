#!/bin/bash

fileid="1Ow6ySJk4msC9Ko464aEeCH21s0X_JgXU"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o checkpoint_pb.tar.gz
tar -zxvf checkpoint_pb.tar.gz
rm checkpoint_pb.tar.gz
rm cookie

echo Download finished.
