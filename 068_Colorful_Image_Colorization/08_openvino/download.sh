#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1cK94Lygsj0P3NTswjotPTXDO-5aiYwU_" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1cK94Lygsj0P3NTswjotPTXDO-5aiYwU_" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
