#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=13acesn4RLw8ajV7-qcG_JVl_5Tn-p-LJ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=13acesn4RLw8ajV7-qcG_JVl_5Tn-p-LJ" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
