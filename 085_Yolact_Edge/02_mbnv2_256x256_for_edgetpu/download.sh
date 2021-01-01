#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1E1bohnqB104PxYHsM27LbnAk_GGFQn9S" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1E1bohnqB104PxYHsM27LbnAk_GGFQn9S" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
