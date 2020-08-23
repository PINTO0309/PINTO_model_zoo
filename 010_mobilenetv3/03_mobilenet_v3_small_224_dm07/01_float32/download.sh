#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1P1tpdn9vmwqw4imSYd2cfKX0aO3H9DJX" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1P1tpdn9vmwqw4imSYd2cfKX0aO3H9DJX" -o v3-small_224_0.75_float.pb

echo Download finished.
