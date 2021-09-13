#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1A_Wr88UcH4R7C_N31d6k9U_gxkvJfV28" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1A_Wr88UcH4R7C_N31d6k9U_gxkvJfV28" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
