#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1oSX_CHJTq5aAEXLuhQMMyZf6qS4_X-IH" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1oSX_CHJTq5aAEXLuhQMMyZf6qS4_X-IH" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
