#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1UvkEpZ_nF2uAl5wMZ8nhgXLuSSuq_eTS" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1UvkEpZ_nF2uAl5wMZ8nhgXLuSSuq_eTS" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
