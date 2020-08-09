#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1BvxOMxkg2ZfK3Y4zlw_DMm5YRRynOvex" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1BvxOMxkg2ZfK3Y4zlw_DMm5YRRynOvex" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
