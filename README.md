# BrainGFM (ICLR 2026)

<div align="center">

<h2>A Brain Graph Foundation Model: Pre-Training and Prompt-Tuning across Broad Atlases and Disorders</h2>

<p>
  Xinxu Wei</strong></a><sup>1</sup>, 
  Kanhao Zhao<sup>1</sup>, 
  Yong Jiao<sup>1</sup>, 
  Lifang He<sup>1</sup>, 
  Yu Zhang<sup>2</sup>
</p>

<p>
  <sup>1</sup>Lehigh University, Bethlehem PA, USA &nbsp;&nbsp;&nbsp;
  <sup>2</sup>Stanford University, Palo Alto CA, USA
</p>

<table>
  <tr>
    <td align="center" width="50%">
      <img src="assets/lehigh_logo.png" width="220"/>
    </td>
    <td align="center" width="50%">
      <img src="assets/stanford_logo.png" width="220"/>
    </td>
  </tr>
</table>

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2506.02044)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey)](#license)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#environment-setup)

</div>

---

## Overview

<!-- ✅ Replace the image filename(s) below with your actual files in ./assets -->
![BrainGFM Overview](assets/Overview.png)

**BrainGFM** is a graph-based brain foundation model for connectome representation learning.
It unifies:
- **Graph Contrastive Learning (GCL)** for view-invariant representation learning
- **Graph Masked Autoencoder (GMAE)** for structure-aware reconstruction



---

## 🏗️ Project Architecture

```text
BrainGFM/
├── assets/                      # Figures used in README
├── checkpoint/                  # Pretrained Checkpoint
├── BrainGFM_pretrain.py         # Core pretraining model (GCL/GMAE)
├── BrainGFM_Gprompt.py          # BrainGFM with graph prompt module
├── graph_prompt.py              # Prompt implementation utilities
├── disease_names.py             # Disease name / mapping
├── utils.py                     # Shared utils
├── main_pretrain.py             # Entry for pretraining
├── main_baseline.py             # Run baseline BrainGFM without pretrained model and finetuning
└── main_finetune.py             # Entry for finetuning / downstream eval

```


## Installation

1. Clone the repository:

```bash
git clone https://github.com/weixinxu666/BrainGFM.git
cd BrainGFM

```


## 🚀 Execution Pipeline

All scripts in this repository are executable independently.  
However, we recommend following the pipeline below for standard usage and reproducibility.

---

### 1️⃣ Pretraining — Brain Foundation Model Learning

Train **BrainGFM** using Graph Contrastive Learning (GCL) and Graph Masked Autoencoder (GMAE):

```bash
python main_pretrain.py
```

This step:

- Performs self-supervised pretraining (GCL / GMAE)
- Learns atlas- and disorder-general representations
- Saves pretrained weights into the `checkpoint/` directory

---

### 2️⃣ Finetuning — Downstream Task Evaluation

After pretraining, run downstream disease classification:

```bash
python main_finetune.py
```

This step:

- Loads pretrained weights from `checkpoint/`
- Performs supervised finetuning
- Supports multi-atlas and multi-disorder settings
- Reports classification performance metrics

---

### 3️⃣ Baseline — Training Without Pretraining

To train BrainGFM **from scratch** (no pretraining):

```bash
python main_baseline.py
```

This is used for comparison with the pretrained model.

---

## 🔄 Recommended Order

```text
Step 1: main_pretrain.py
Step 2: main_finetune.py
Step 3: main_baseline.py (for comparison and ablation studies)
```

---

This pipeline ensures consistent experimental settings and fair evaluation across pretrained and non-pretrained models.


## Pretrained Checkpoint

We provide a pretrained model checkpoint to facilitate reproducibility and further research.

The checkpoint contains the trained model parameters used in our experiments and can be directly loaded for evaluation, fine-tuning, or downstream tasks.

**Download link:**  
[Checkpoint Model](https://drive.google.com/file/d/1uWlfDqT37kc6jKQ21xwnKvy8NSqbf2B2/view?usp=drive_link)




## Citation

```
@article{wei2025braingfm,
  title   = {A Brain Graph Foundation Model: Pre-Training and Prompt-Tuning for Any Atlas and Disorder},
  author  = {Wei, Xinxu and Zhao, Kanhao and Jiao, Yong and He, Lifang and Zhang, Yu},
  journal = {arXiv preprint arXiv:2506.02044},
  year    = {2025}
}
```


## Contact

For questions or collaborations, please contact:  
[xiw523@lehigh.edu](mailto:xiw523@lehigh.edu)
