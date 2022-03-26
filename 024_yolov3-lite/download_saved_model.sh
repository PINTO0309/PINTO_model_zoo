#!/bin/bash

fileid="1byobKK7kfHuKumnada7rDSD1l2x9d6xO"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o saved_model.tar.gz
tar -zxvf saved_model.tar.gz
rm saved_model.tar.gz
rm cookie

echo Download finished.
