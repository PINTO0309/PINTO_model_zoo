#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=14ShuvvXj1S4fX5is2SLH2vH-0ZhLPg9-" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=14ShuvvXj1S4fX5is2SLH2vH-0ZhLPg9-" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
