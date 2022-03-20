https://user-images.githubusercontent.com/33194443/159169372-126cc023-4fac-4e69-8efe-a2df8ca8c1c2.mp4

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

# Sample
![image](https://user-images.githubusercontent.com/33194443/159132347-98d03023-1ca2-499d-9f71-83144f1e9214.png)
![image](https://user-images.githubusercontent.com/33194443/159132287-6c7aa9fb-ad23-486f-8ca7-9d7f3d1a954d.png)
![image](https://user-images.githubusercontent.com/33194443/159132357-bcb7850d-e34e-4676-9285-3153fc39092c.png)
![image](https://user-images.githubusercontent.com/33194443/159132364-2b194b1f-9d3a-4e0d-8152-12fcdb57ef43.png)
