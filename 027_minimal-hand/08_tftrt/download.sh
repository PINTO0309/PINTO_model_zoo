#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1HCg51JJwq5HxJ1JV1oih5j81R1L4wB2l" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1HCg51JJwq5HxJ1JV1oih5j81R1L4wB2l" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
