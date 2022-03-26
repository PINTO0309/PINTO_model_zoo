#!/bin/bash

fileid="12FKxTMcz65I-RmY9yDvj0IIr2oP8Hars"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o tflite_graph_with_postprocess.pb

echo Download finished.
