#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1cFgcmYnp_LLzEbthX6CDQ_QPSJT0Up9i" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1cFgcmYnp_LLzEbthX6CDQ_QPSJT0Up9i" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
