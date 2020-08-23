#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1O4Qn2id8quh8wfjuBHzH6LRC_DgrJnnz" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1O4Qn2id8quh8wfjuBHzH6LRC_DgrJnnz" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
