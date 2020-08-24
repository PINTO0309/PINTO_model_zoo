#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1JBpaC7u5b-uejJhJ7eCnJAGM63k_JF20" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1JBpaC7u5b-uejJhJ7eCnJAGM63k_JF20" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
