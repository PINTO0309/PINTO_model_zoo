# Demo projects

## SparseInst with ONNX Runtime in Python
```
python demo_SparseInst_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default=
        'sparseinst_r50_giam_aug_768x1344/sparseinst_r50_giam_aug_768x1344_opt.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='768,1344',
    )
```
