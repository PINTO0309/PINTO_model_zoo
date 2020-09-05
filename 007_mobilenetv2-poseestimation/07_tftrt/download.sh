#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1-I29Olp7XSyNIpQ4qAvPXSQUTq48MZyl" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1-I29Olp7XSyNIpQ4qAvPXSQUTq48MZyl" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
