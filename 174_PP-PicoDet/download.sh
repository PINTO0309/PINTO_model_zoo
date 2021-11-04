#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=16O5h8Wf9mPSLtBAvUSyuLZ4f0MNiSDw-" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=16O5h8Wf9mPSLtBAvUSyuLZ4f0MNiSDw-" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
