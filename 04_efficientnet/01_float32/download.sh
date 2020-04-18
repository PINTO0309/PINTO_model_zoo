#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1d4G1DsoEAeam9RD7597bDt7Ex-7eIeiK" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1d4G1DsoEAeam9RD7597bDt7Ex-7eIeiK" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
