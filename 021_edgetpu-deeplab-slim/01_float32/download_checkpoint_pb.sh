#!/bin/bash

fileid="1Rrfe20sd3m00lfWVaK2P7EBu8piF81f6"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o checkpoint_pb.tar.gz
tar -zxvf checkpoint_pb.tar.gz
rm checkpoint_pb.tar.gz
rm cookie

echo Download finished.
