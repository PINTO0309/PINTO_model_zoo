# Demo projects

## CoCosNet with ONNX Runtime in Python
This sample only works with "cocosnet.onnx".<br>
```
python demo_CoCosNet(only_cocosnet_onnx).py
```

If you want to change the input image and segmentation map, specify the following arguments.<br>
```python
    parser.add_argument(
        "--input_seg_map",
        type=str,
        default='image/input_seg_map/2.png',
    )
    parser.add_argument(
        "--ref_image",
        type=str,
        default='image/ref_image/1.jpg',
    )
    parser.add_argument(
        "--ref_seg_map",
        type=str,
        default='image/ref_seg_map/1.png',
    )
```

## About sample images
The sample image uses the image of "[PAKUTASO](https://www.pakutaso.com/)".<br>
If you want to use the image itself for another purpose, you must follow the [userpolicy of PAKUTASO](https://www.pakutaso.com/userpolicy.html).
