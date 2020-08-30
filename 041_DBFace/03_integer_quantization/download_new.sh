#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1CKkLLt2oNW0Tpk4WYVx3-JAz4ibCJPI8" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1CKkLLt2oNW0Tpk4WYVx3-JAz4ibCJPI8" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
