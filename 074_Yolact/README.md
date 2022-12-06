## Note

- YOLACT (Resnet101-FPN) - yolact_base_54_800000_550x550.onnx - MultiClass-NMS + Post-Process

  https://user-images.githubusercontent.com/33194443/205485359-4f4d3423-6502-4de0-bc19-8d11f0185cc4.mp4

- Generate Anchor sample code
  ```python
  def decode(self, loc, priors):
      """
      Decode predicted bbox coordinates using the same scheme
      employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf
          b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
          b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
          b_w = prior_w * exp(loc_w)
          b_h = prior_h * exp(loc_h)
      Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
      while priors are inputed as [x, y, w, h] where each coordinate
      is relative to size of the image (even sigmoid(x)). We do this
      in the network by dividing by the 'cell size', which is just
      the size of the convouts.
      Also note that prior_x and prior_y are center coordinates which
      is why we have to subtract .5 from sigmoid(pred_x and pred_y).
      Args:
          - loc:    The predicted bounding boxes of size [num_priors, 4]
          - priors: The priorbox coords with size [num_priors, 4]
      Returns: A tensor of decoded relative coordinates in point form
              form with size [num_priors, 4]
      """
      priors = priors[np.newaxis, ...]
      variances = [0.1, 0.2]

      boxes = torch.cat(
          (
              priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
              priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1])
          ),
          2
      )

      boxes0 = boxes[:, :, 0] - boxes[:, :, 2] / 2
      boxes1 = boxes[:, :, 1] - boxes[:, :, 3] / 2
      boxes2 = boxes[:, :, 0] + boxes[:, :, 2] / 2
      boxes3 = boxes[:, :, 1] + boxes[:, :, 3] / 2
      boxes = torch.cat(
          [
              boxes0[...,np.newaxis],
              boxes1[...,np.newaxis],
              boxes2[...,np.newaxis],
              boxes3[...,np.newaxis]
          ], dim=2)

      return boxes
  ```

- Post-Process

  https://github.com/PINTO0309/components_of_onnx/tree/main/components_of_onnx/ops/Z11_YOLACT_PostProcess
  ![image](https://user-images.githubusercontent.com/33194443/205787867-2ba358fe-ee97-4dc6-8577-85d99125b0a4.png)
