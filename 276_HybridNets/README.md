# Note

```python
hybridnets_{H}x{W}/anchors_{H}x{W}.npy = anchors

$ python3
>>> import numpy as np
>>> np.load('anchors_256x384.npy')
array([[[  -3.9      ,    0.9      ,   11.9      ,    7.1      ],
        [  -1.       ,   -1.       ,    9.       ,    9.       ],
        [   0.9      ,   -3.9      ,    7.1      ,   11.9      ],
        ...,
        [-123.577965 ,  196.1656   ,  507.57797  ,  443.83438  ],
        [  -7.7328877,  120.26711  ,  391.73288  ,  519.7329   ],
        [  68.16561  ,    4.422037 ,  315.83438  ,  635.57794  ]]],
      dtype=float32)
>>> np.load('anchors_256x384.npy').shape
(1, 18414, 4)
```