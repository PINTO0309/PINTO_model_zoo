# Demo projects

## ShadowFormer with ONNX Runtime in Python
```
python demo_ShadowFormer_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        '--image',
        type=str,
        default='sample.jpg',
    )
    parser.add_argument(
        '--mask',
        type=str,
        default='sample_mask.jpg',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='shadowformer_istd_plus_480x640.onnx',
    )
```

![image](https://user-images.githubusercontent.com/37477845/222126922-5f81b96a-8a8a-41b1-be31-3a9218bb2ea3.png)


## About sample images
The sample image uses the image of "[PAKUTASO](https://www.pakutaso.com/)".<br>
If you want to use the image itself for another purpose, you must follow the [userpolicy of PAKUTASO](https://www.pakutaso.com/userpolicy.html).
