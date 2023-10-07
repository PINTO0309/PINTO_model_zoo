# Note

The absence of ultra-close range annotations in the training dataset results in a significant decrease in recognition rate at close distances of a few tens of centimeters. Instead, it detects very strongly at medium to long range. It is much more resistant to blurring than past lightweight models. There is a bit too much over-detection, but it would be possible to exclude useless detection results when combined with the lightweight and robust skeletal detection of the bottom-up approach.

```
USB Camera: 640x480
Score threshold: 0.50
IoU threshold: 0.05
```

https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/136b892d-a59b-4493-9b99-02d40609db49

