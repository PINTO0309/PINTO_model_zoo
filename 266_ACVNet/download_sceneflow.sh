#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1cAcRd0ifX-BkyvK5K6VxCa-CHtjm26GS" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1cAcRd0ifX-BkyvK5K6VxCa-CHtjm26GS" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
