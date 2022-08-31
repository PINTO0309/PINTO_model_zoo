# Note
![example](https://user-images.githubusercontent.com/33194443/187728847-e3631823-255b-4147-9562-168e5495acac.jpg)

- `model_105_anchor_grid.npy`
  - torch.Size([3, 3, 1, 1, 2])
    - anchor_grid0 torch.Size([1, 3, 1, 1, 2])
    - anchor_grid1 torch.Size([1, 3, 1, 1, 2])
    - anchor_grid2 torch.Size([1, 3, 1, 1, 2])

```python
import numpy as np

anchor_grids = np.load('model_105_anchor_grid.npy')
anchor_grids.shape
(3, 1, 3, 1, 1, 2)

anchor_grid0 = anchor_grids[0]
anchor_grid1 = anchor_grids[1]
anchor_grid2 = anchor_grids[2]

anchor_grid0.shape
(1, 3, 1, 1, 2)
anchor_grid1.shape
(1, 3, 1, 1, 2)
anchor_grid2.shape
(1, 3, 1, 1, 2)
```
