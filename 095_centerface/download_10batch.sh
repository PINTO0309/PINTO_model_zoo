#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=12oE1zHvRBisLe15Y08KYmHkO9rXdaGQP" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=12oE1zHvRBisLe15Y08KYmHkO9rXdaGQP" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
