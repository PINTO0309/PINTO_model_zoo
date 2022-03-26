#!/bin/bash

fileid="122zfOxnmWSqkDVAwQUYI4ZQOjtmKB1tg"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o deeplab_mnv3_small_cityscapes_trainfine_2019_11_15.tar.gz
tar -zxvf deeplab_mnv3_small_cityscapes_trainfine_2019_11_15.tar.gz
rm deeplab_mnv3_small_cityscapes_trainfine_2019_11_15.tar.gz

echo Download finished.