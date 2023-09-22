# Movement Primitive - Transformer


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
![Tests](https://github.com/rasdani/mp-transformer/actions/workflows/tests.yml/badge.svg)


This transformer architecture maps a sequence of pose joint angles, i.e. frames of movement, to a
(smaller) sequence of latent primitives.
These act as low-dimensional building blocks of movement and are encoded as distributions in latent space.
It then samples from these like a VAE and feeds them to a sequence decoder.
After applying gaussian masks, the subsequences are averaged
to form the final reconstructed movement sequence.
Adapted from [Marsot et. al. (2022)](https://arxiv.org/abs/2206.13142) \[[code](https://gitlab.inria.fr/mmarsot/new_segmentation)\].

The goal is to explore a new model for [Movement Primitives](https://www.mdpi.com/1099-4300/20/10/724) trained on data of a simulated toy limb.

Potential use cases are modeling Movement Primitives in our VR lab or in the wild.

## Setup
Install with `mamba env create --file environment.yml` (tested with mamba and [mambaforge](https://github.com/conda-forge/miniforge#mambaforge) on Ubuntu 22.04., conda should work aswell albeit slowly.)
Make sure `ffmpeg` is installed on your system for rendering videos.

Activate the environment with `mamba activate mp-transformer` and setup with `pip install -e .`.

## Run
Activate the environment and run training without logging to Weights & Biases with `python mp_transformer/train.py --no-log`.

## Demo
Example videos and Jupyter notebooks can be found in `demo/example.mp4`.
