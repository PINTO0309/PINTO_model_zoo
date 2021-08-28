#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Gng3s4W_nm5Awq3Z3K19gsWrmuNUdIq8" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Gng3s4W_nm5Awq3Z3K19gsWrmuNUdIq8" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
