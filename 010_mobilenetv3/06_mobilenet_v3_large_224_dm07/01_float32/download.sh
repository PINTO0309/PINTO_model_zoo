#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1gL7TzJOWAckfwMJKQj8H9t0mOVakKfm7" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1gL7TzJOWAckfwMJKQj8H9t0mOVakKfm7" -o v3-large_224_0.75_float.pb

echo Download finished.
