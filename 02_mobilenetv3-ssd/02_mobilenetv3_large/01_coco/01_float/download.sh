#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1mgSm5uDAijh3pniX-cPzWHM79Lmz4Hw2" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1mgSm5uDAijh3pniX-cPzWHM79Lmz4Hw2" -o frozen_inference_graph.pb

echo Download finished.
