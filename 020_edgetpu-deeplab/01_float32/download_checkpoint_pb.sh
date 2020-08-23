#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=11vCAygH0YBTS42E5N4l7RxsguyN08CkH" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=11vCAygH0YBTS42E5N4l7RxsguyN08CkH" -o checkpoint_pb.tar.gz
tar -zxvf checkpoint_pb.tar.gz
rm checkpoint_pb.tar.gz
echo Download finished.
