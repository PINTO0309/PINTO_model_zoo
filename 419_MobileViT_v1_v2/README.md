# Refference
- https://github.com/apple/ml-cvnets
- https://github.com/apple/ml-cvnets/blob/main/docs/source/en/general/README-model-zoo.md#mobilevitv1-legacy
- https://github.com/apple/ml-cvnets/blob/main/docs/source/en/general/README-model-zoo.md#mobilevitv2-256x256
- https://github.com/apple/ml-cvnets/blob/main/docs/source/en/general/README-pytorch-to-coreml.md

Fork repository
- https://github.com/NobuoTsukamoto/ml-cvnets

# Licence
- https://github.com/apple/ml-cvnets/blob/main/LICENSE

# How to

## torch -> onnx

Note: I used podman, but if you are using docker, please read podman as docker.
```
$ git clone https://github.com/NobuoTsukamoto/ml-cvnets.git
$ podman run --rm -it -v `pwd`/ml-cvnets:/ml-cvnets pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
# apt update && apt install -y wget
# cd /ml-cvnets
# pip install -r requirements.txt -c constraints.txt
# pip install --editable .
# export PYTHONPATH=.
# ./export_torch.sh
# exit
```

onnx -> onnx2tf -> tf2onnx -> spo4onnx
```
$ podman run --rm -it -v `pwd`/ml-cvnets:/ml-cvnets docker.io/pinto0309/onnx2tf:1.16.30

$ pip install tf2onnx tqdm
$ cd /ml-cvnets/coreml_models_cls/
$ cp ../convert_tf2onnx_onnx2tf.sh ./
$ ./convert_tf2onnx_onnx2tf.sh
```
