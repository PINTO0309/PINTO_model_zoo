#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1UiboYzQQWWUuKAj2al_zQ3u5Bil8H-oa" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=147mG1UiboYzQQWWUuKAj2al_zQ3u5Bil8H-oa" -o deeplab_mnv3_large_cityscapes_trainfine_769.pb

echo Download finished.
