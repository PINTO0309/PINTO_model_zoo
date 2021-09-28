#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1lhz4emW5Vdpg7n7K7DwabDYDkZBHJJoQ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1lhz4emW5Vdpg7n7K7DwabDYDkZBHJJoQ" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
