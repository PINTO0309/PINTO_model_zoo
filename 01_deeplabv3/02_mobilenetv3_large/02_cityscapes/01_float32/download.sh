#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1QxS3G55rUQvuiBF-hztQv5zCkfPfwlVU" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1QxS3G55rUQvuiBF-hztQv5zCkfPfwlVU" -o deeplab_mnv3_large_cityscapes_trainfine_2019_11_15.tar.gz
tar -zxvf deeplab_mnv3_large_cityscapes_trainfine_2019_11_15.tar.gz
rm deeplab_mnv3_large_cityscapes_trainfine_2019_11_15.tar.gz

echo Download finished.
