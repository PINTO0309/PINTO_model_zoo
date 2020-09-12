#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1J5h1JJ5yOKu0Ex_TMaW1hGgPVo3Lb2YC" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1J5h1JJ5yOKu0Ex_TMaW1hGgPVo3Lb2YC" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
