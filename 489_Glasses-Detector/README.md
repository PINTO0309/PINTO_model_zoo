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

## 1. ONNX test
  - Installation
    https://github.com/PINTO0309/glasses-detector#installation
  - Demonstration of models
    ```
    uv run python demo/demo_classification_sunglasses.py --camera 0
    ```

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
