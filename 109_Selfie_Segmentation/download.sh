#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=14xJrbB9S7kGcq8dkeI39z6W2W9axDFIq" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=14xJrbB9S7kGcq8dkeI39z6W2W9axDFIq" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
