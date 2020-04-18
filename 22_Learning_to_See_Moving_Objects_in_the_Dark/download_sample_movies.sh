#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1kxtTDg9H50Onkzgv-mvt8n5hv69lxziW" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1kxtTDg9H50Onkzgv-mvt8n5hv69lxziW" -o sample_movies.tar.gz
tar -zxvf sample_movies.tar.gz
rm sample_movies.tar.gz
echo Download finished.
