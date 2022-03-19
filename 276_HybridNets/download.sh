#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1r1jDtJhi-5KfyMOee0CEbAC-N_hmf8t7" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1r1jDtJhi-5KfyMOee0CEbAC-N_hmf8t7" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
