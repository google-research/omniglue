# \[CVPR'24\] Code release for OmniGlue

[Hanwen Jiang](https://hwjiang1510.github.io/),
[Arjun Karpur](https://scholar.google.com/citations?user=jgSItF4AAAAJ),
[Bingyi Cao](https://scholar.google.com/citations?user=7EeSOcgAAAAJ),
[Qixing Huang](https://www.cs.utexas.edu/~huangqx/),
[Andre Araujo](https://andrefaraujo.github.io/)

--------------------------------------------------------------------------------

[**Project Page**](https://hwjiang1510.github.io/OmniGlue/) | [**Paper**]() |
[**Usage**]()

Official code release for the CVPR 2024 paper: **OmniGlue: Generalizable Feature
Matching with Foundation Model Guidance**.

**Abstract:** The image matching field has been witnessing a continuous
emergence of novel learnable feature matching techniques, with ever-improving
performance on conventional benchmarks. However, our investigation shows that
despite these gains, their potential for real-world applications is restricted
by their limited generalization capabilities to novel image domains. In this
paper, we introduce OmniGlue, the first learnable image matcher that is designed
with generalization as a core principle. OmniGlue leverages broad knowledge from
a vision foundation model to guide the feature matching process, boosting
generalization to domains not seen at training time. Additionally, we propose a
novel keypoint position-guided attention mechanism which disentangles spatial
and appearance information, leading to enhanced matching descriptors. We perform
comprehensive experiments on a suite of 6 datasets with varied image domains,
including scene-level, object-centric and aerial images. OmniGlue’s novel
components lead to relative gains on unseen domains of 18.8% with respect to a
directly comparable reference model, while also outperforming the recent
LightGlue method by 10.1% relatively.


## Installation

First, use pip to install `omniglue`:

```sh
conda create -n omniglue
conda activate omniglue

git clone <..repo...>
cd omniglue
pip install -e .
```

Then , download the following models to `./models/`

-   [SuperPoint weights]()
-   [DINOv2 weights]()
-   [OmniGlue weights]()

```sh
mkdir models
mv ~/Downloads/og_export ~/Downloads/sp_v6  ~/Downloads/dinov2_vitb14.pth models/
```

## Usage

```py

import omniglue

image0 = ... # load images from file into np.array
image1 = ...

og = omniglue.OmniGlue(
  og_export='./models/og_export',
  sp_export='./models/sp_v6',
  dino_export='./models/dinov2_vitb14.pth',
)

match_kp0s, match_kp1s, match_confidences = og.FindMatches(image0, image1)
# Output:
#   match_kp0: (N, 2) array of (x,y) coordinates in image0.
#   match_kp1: (N, 2) array of (x,y) coordinates in image1.
#   match_confidences: N-dim array of each of the N match confidence scores.
```

## Demo

`demo.py` contains example usage of the `omniglue` module.

```sh
conda activate omniglue
python demo.py
# <see output in './demo_output.png'>
```

## Citation

If you use our model for your project, please cite: `TODO`

--------------------------------------------------------------------------------

This is not an officially supported Google product.
