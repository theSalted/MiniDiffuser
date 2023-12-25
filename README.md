# MiniDiffuser

MiniDiffuser is a mini diffusion model based on [IADB](https://github.com/tchambon/IADB) (See [paper](https://arxiv.org/abs/2305.03486), [blog post](https://ggx-research.github.io/publication/2023/05/10/publication-iadb.html) and, [2D tutorial](https://tchambon.github.io/posts/iadb-2Da/)).
<br />

## Quick start

If you want to setup a new conda environment, download a dataset (celeba) and launch a training, you can follow this:

```shell
conda env create -f environment.yml
conda activate minid
python3 minid.py
```

Replace `python3 minid.py` with `python3 minid.py --e 10` for a quicker training process (this set epoch size to 10 instead the default 100).

### Exit

Press `crt+c` to exit training.

To exit your environment run:

```shell
conda deactivate
```

### Help

To see all features flags use:

```shell
python3 minid.py -h
```

Mini Diffuser currently support `celeba` and `cifar10` (default) as dataset, modify `minid.py` to add yours.

### Update you environment

To update your environment run:

```shell
conda env update -f environment.yml
```

## Setup

Python 3 dependencies:

- [Pytorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/)
- [Diffusers](https://github.com/huggingface/diffusers)

This code has been tested with Python 3.8 on Ubuntu 22.04. We recommend setting up a dedicated Conda environment using Python 3.8 and Pytorch 2.0.1.

## Code description

The iadb.py contains a simple training loop.

It demonstrates how to train a new MiniDiffuser model and how to generate results.
