#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1IQIaKwQ61RQCDAQuNmLHh0fqzRw8wEVT" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1IQIaKwQ61RQCDAQuNmLHh0fqzRw8wEVT" -o checkpoint.tar.gz
tar -zxvf checkpoint.tar.gz
rm checkpoint.tar.gz
echo Download finished.
