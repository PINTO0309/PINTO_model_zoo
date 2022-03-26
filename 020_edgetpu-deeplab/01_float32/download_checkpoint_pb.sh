#!/bin/bash
# 404 Not found ?
fileid="11vCAygH0YBTS42E5N4l7RxsguyN08CkH"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o checkpoint_pb.tar.gz
tar -zxvf checkpoint_pb.tar.gz
rm checkpoint_pb.tar.gz
rm cookie

echo Download finished.
