#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1yxr4L_Z0T15r-qcS1ywOaM2geSHFw_YR" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1yxr4L_Z0T15r-qcS1ywOaM2geSHFw_YR" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
