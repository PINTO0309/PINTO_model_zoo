#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1v3v2k9ZZjLd6Fzx1_sioc-l4BwminsVC" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1v3v2k9ZZjLd6Fzx1_sioc-l4BwminsVC" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
