#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1VVo1FxoZCAgAWWsmaf3EG45CGS8IXmnS" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1VVo1FxoZCAgAWWsmaf3EG45CGS8IXmnS" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
