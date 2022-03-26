#!/bin/bash

fileid="1bTVIKWJaSNWHch1L1FLIQOd3Xo7KUujG"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ssdlite_mobilenet_v2_voc_2020_02_04.tar.gz
tar -zxvf ssdlite_mobilenet_v2_voc_2020_02_04.tar.gz
rm ssdlite_mobilenet_v2_voc_2020_02_04.tar.gz

echo Download finished.
