#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=160P9M7SmUpwzzs3a7hrRWRSQWuZELu86" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=160P9M7SmUpwzzs3a7hrRWRSQWuZELu86" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
