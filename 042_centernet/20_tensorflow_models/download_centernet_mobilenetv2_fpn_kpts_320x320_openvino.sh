#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=11vgmgtK0d-A-kMkOGSLA51_3b3FqYREF" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=11vgmgtK0d-A-kMkOGSLA51_3b3FqYREF" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
