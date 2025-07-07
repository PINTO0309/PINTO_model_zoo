# 469_Face_Deblurring

![image](https://github.com/user-attachments/assets/f28a1a1b-d566-4660-8253-ba69a29c1f23)

```python
deblurred_images = np.clip(deblurred_images, 0, 255)
f, ax = plt.subplots(3,10, figsize=(15,5))
for i in range(10):
    ax[0,i].imshow(Images[i].astype('uint8'))
    ax[0,i].axis('Off')
    ax[0,i].set_title('Clean', size=15)
    ax[1,i].imshow(Blurry[i].astype('uint8'))
    ax[1,i].axis('Off')
    ax[1,i].set_title('Blurry', size=15)
    ax[2,i].imshow(deblurred_images[i].astype('uint8'))
    ax[2,i].axis('Off')
    ax[2,i].set_title('Deblurred', size=15)
plt.show()
```

https://github.com/PINTO0309/Face-Blurring-and-Deblurring

https://github.com/PINTO0309/Face-Blurring-and-Deblurring/blob/onnx/model/faceDeblurring/Face_Deblurring.ipynb

## Cited

https://github.com/hrugved06/Face-Blurring-and-Deblurring
