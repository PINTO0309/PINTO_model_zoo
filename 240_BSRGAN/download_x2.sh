#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=16--j7Beq_tU0hkSLN3pO-gQN8JNebJDa" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=16--j7Beq_tU0hkSLN3pO-gQN8JNebJDa" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
