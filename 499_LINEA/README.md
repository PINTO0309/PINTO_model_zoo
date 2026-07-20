# 499_LINEA

LINEA: Fast and accurate line detection using scalable transformers

<p align="center">
  <a href="https://github.com/SebastianJanampa/LINEA/master/LICENSE">
        <img alt="colab" src="https://img.shields.io/badge/license-apache%202.0-blue?style=for-the-badge">
  </a>

  <a href="https://arxiv.org/abs/2505.16264">
        <img alt="arxiv" src="https://img.shields.io/badge/-paper-gray?style=for-the-badge&logo=arxiv&labelColor=red">
  </a>

  <a href="https://colab.research.google.com/github/SebastianJanampa/LINEA/blob/master/LINEA_tutorial.ipynb">
        <img alt="colab" src="https://img.shields.io/badge/-colab-blue?style=for-the-badge&logo=googlecolab&logoColor=white&labelColor=%23daa204&color=yellow">
  </a>

  <a href='https://huggingface.co/spaces/SebasJanampa/LINEA'>
      <img src='https://img.shields.io/badge/-SPACE-orange?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=FF5500&color=orange'>
   </a>

</p>

<p align="center">
    <a href="https://arxiv.org/abs/2505.16264">LINEA: Fast and accurate line detection using scalable transformers</a>
</p>


<p align="center">
Sebastian Janampa and Marios Pattichis
</p>

<p align="center">
The University of New Mexico
  <br>
Department of Electrical and Computer Engineering
</p>

<p align="center">
  <a href="https://paperswithcode.com/sota/line-segment-detection-on-york-urban-dataset?p=linea-fast-and-accurate-line-detection-using">
    <img alt="sota" src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/linea-fast-and-accurate-line-detection-using/line-segment-detection-on-york-urban-dataset">
    </a>
</p>

<p align="center">
  <img src=https://github.com/user-attachments/assets/21f9fc86-d0bb-4ebe-9d82-e68991818bd8 border=0 width=1000>
</p>

LINEA is a powerful real-time line detector that introduces Line Attention mechanism,
achieving outstanding performance without being pretrained on COCO or Object365 datasets.

<details open>
<summary> Attention Mechanishm </summary>

We compare line attention with traditional attention and deformable attention.
We highlight two advantages of our proposed mechanism:

- Line attention is a sparse mechanism like deformable attention. This significantly reduces memory complexity.
- Line attention pays attention to the line endpoints like traditional attention but also attends locations between the endpoints.

<p align="center">
  <img src=https://github.com/user-attachments/assets/a08cb164-458d-4802-abdc-0294864c89d4 border=0 width=1000>
</p>

</details>

https://github.com/user-attachments/assets/14b4b9cf-6378-4482-b06f-d1af71fb5aa7

## Original

https://github.com/SebastianJanampa/LINEA

## ONNX demo

https://github.com/PINTO0309/LINEA

## Citation
If you use `LINEA` or its methods in your work, please cite the following BibTeX entries:
<details open>
<summary> bibtex </summary>

```bibtex
@misc{janampa2025linea,
      title={LINEA: Fast and Accurate Line Detection Using Scalable Transformers},
      author={Sebastian Janampa and Marios Pattichis},
      year={2025},
      eprint={2505.16264},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.16264},
}
```
</details>
