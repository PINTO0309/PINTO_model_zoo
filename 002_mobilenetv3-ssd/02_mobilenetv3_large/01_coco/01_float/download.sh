#!/bin/bash

fileid="1mgSm5uDAijh3pniX-cPzWHM79Lmz4Hw2"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o frozen_inference_graph.pb

echo Download finished.
