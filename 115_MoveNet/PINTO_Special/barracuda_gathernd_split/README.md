## Sample
- e.g. Make **`GatherND`**
  ```bash
  python make_gathernd_replace.py \
  --model_name_suffix 0 \
  --data_shape 1 48 64 17 \
  --indices_shape 6 3 \
  --opset 11
  ```
  ![image](https://user-images.githubusercontent.com/33194443/192176507-315edf75-4975-400f-a623-74517ddbeb70.png)

- e.g. Make **`Split`**
  ```bash
  python make_split_replace.py \
  --opset 11 \
  --model_name_suffix 4 \
  --data_shape 102 2 \
  --data_type float32 \
  --split_axis 1 \
  --split_number_of_after_division 1
  ```
  ![image](https://user-images.githubusercontent.com/33194443/192176576-e786a554-d0fb-470d-8771-8883cb660508.png)
  ![image](https://user-images.githubusercontent.com/33194443/192176591-e3e85a32-7f0a-43bf-a0fb-16de980db68e.png)

## References
1. [GatherND implementation and ONNX export for Unity Barracuda](https://zenn.dev/pinto0309/scraps/d6598463fd75e2)
2. [Split implementation and ONNX export for Unity Barracuda](https://zenn.dev/pinto0309/scraps/69fa6d74bb3de3)
3. [simple-onnx-processing-tools](https://github.com/PINTO0309/simple-onnx-processing-tools)
