#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1jeLn-d1fPn3sm3TZfPr1np48N_otVVv3" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1jeLn-d1fPn3sm3TZfPr1np48N_otVVv3" -o efficientnet-lite4.tar.gz
tar -zxvf efficientnet-lite4.tar.gz
rm efficientnet-lite4.tar.gz
echo Download finished.
