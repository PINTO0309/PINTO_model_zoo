#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Q5sefAWnDMWJ4K28y_MkeFebwwy1wLYL" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Q5sefAWnDMWJ4K28y_MkeFebwwy1wLYL" -o sample_movies.tar.gz
tar -zxvf sample_movies.tar.gz
rm sample_movies.tar.gz
echo Download finished.