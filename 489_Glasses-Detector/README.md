# 489_Glasses-Detector

Package for processing images with different types of glasses and their parts. It provides a quick way to use the pre-trained models for **3** kinds of tasks, each divided into multiple categories, for instance, *classification of sunglasses* or *segmentation of glasses frames*.

<br>

<div align="center">

<table align="center"><tbody>
    <tr><td><strong>Classification</string></td> <td> 👓 <em>transparent</em> 🕶️ <em>opaque</em> 🥽 <em>any</em> ➿<em>shadows</em></td></tr>
    <tr><td><strong>Detection</string></td> <td> 🤓 <em>worn</em> 👓  <em>standalone</em> 👀 <em>eye-area</em></td></tr>
    <tr><td><strong>Segmentation</string></td> <td> 😎 <em>full</em> 🖼️ <em>frames</em> 🦿 <em>legs</em> 🔍 <em>lenses</em> 👥 <em>shadows</em></td></tr>
</tbody></table>

$\color{gray}{\textit{Note: }\text{refer to}}$ [Glasses Detector Features](https://mantasu.github.io/glasses-detector/docs/features.html) $\color{gray}{\text{for visual examples.}}$

</div>

1. classification
    1. anyglasses      # Datasets with any glasses as positives
    2. eyeglasses      # Datasets with transparent glasses as positives
    3. shadows         # Datasets with visible glasses frames shadows as positives
    4. sunglasses      # Datasets with semi-transparent/opaque glasses as positives
2. detection
    1. eyes            # Datasets with bounding boxes for eye area
    2. solo            # Datasets with bounding boxes for standalone glasses
    3. worn            # Datasets with bounding boxes for worn glasses
3. segmentation
    1. frames          # Datasets with masks for glasses frames
    2. full            # Datasets with masks for full glasses (frames + lenses)
    3. legs            # Datasets with masks for glasses legs (part of frames)
    4. lenses          # Datasets with masks for glasses lenses
    5. shadows         # Datasets with masks for eyeglasses frames cast shadows
    6. smart           # Datasets with masks for glasses frames and lenses if opaque

## 1. ONNX test
  - Installation
    https://github.com/PINTO0309/glasses-detector#installation
  - Demonstration of models
    ```
    uv run python demo/demo_classification_sunglasses.py --camera 0
    ```

    https://github.com/user-attachments/assets/852af769-ffae-4650-9429-35d8a748b055

## 2. Cited
  I am very grateful for their excellent work.
  - glasses-detector

    https://github.com/mantasu/glasses-detector

    ```bibtex
    @software{Birskus_Glasses_Detector_2024,
        author = {Birškus, Mantas},
        title = {{Glasses Detector}},
        license = {MIT},
        url = {https://github.com/mantasu/glasses-detector},
        month = {3},
        year = {2024},
        doi = {10.5281/zenodo.8126101}
    }
    ```
