#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=10kPuIVzrJlW9P_qc4LQ2epCfgN8Kgn6x" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=10kPuIVzrJlW9P_qc4LQ2epCfgN8Kgn6x" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
