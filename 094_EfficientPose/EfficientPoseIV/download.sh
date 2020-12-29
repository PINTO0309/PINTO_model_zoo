#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1XqnH429euhs4w6qL9GJ_zcWql-sjxqMn" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1XqnH429euhs4w6qL9GJ_zcWql-sjxqMn" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
