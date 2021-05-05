#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1oCOqZ7AK7ja-R_Jphpc-H1skbodd2RQr" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1oCOqZ7AK7ja-R_Jphpc-H1skbodd2RQr" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
