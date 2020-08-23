#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1DVadju5YcRQzS5JzAFCFxdgwDh6-sifv" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1DVadju5YcRQzS5JzAFCFxdgwDh6-sifv" -o weights.pb

echo Download finished.
