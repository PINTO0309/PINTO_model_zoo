#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1NgaBdkqoT8gjiVjMf6hHZqzEK2Es-e9w" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1NgaBdkqoT8gjiVjMf6hHZqzEK2Es-e9w" -o mobilenet_v2_1.0_224_frozen.pb

echo Download finished.
