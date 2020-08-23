#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1pLHMKmWdO3PuIF1so9SRhZ9Wdjr3gua-" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1pLHMKmWdO3PuIF1so9SRhZ9Wdjr3gua-" -o v3-small_224_1.0_float.pb

echo Download finished.
