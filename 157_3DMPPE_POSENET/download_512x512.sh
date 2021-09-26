#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1vpbHxFLPbuPym2M_xQz8Bzgnz6NjBOeX" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1vpbHxFLPbuPym2M_xQz8Bzgnz6NjBOeX" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
