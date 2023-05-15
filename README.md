# Movement Primitive - Transformer


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
![Tests](https://github.com/rasdani/actions/workflows/tests.yml/badge.svg)


This transformer architecture maps a sequence of poses, i.e. frames of movement, to a
(smaller) sequence of latent primitives.
These act as low-dimensional building blocks of movement and are encoded as distributions in latent space.
It then samples from these like a VAE and feeds them to a sequence decoder.
After applying gaussian masks, the subsequences are averaged
to form the final reconstructed movement sequence.
Adapted from [Marsot et. al. (2022)](https://arxiv.org/abs/2206.13142) \[[code](https://gitlab.inria.fr/mmarsot/new_segmentation)\].

The goal is to explore a new model for [Movement Primitives](https://www.mdpi.com/1099-4300/20/10/724) trained on [Human3.6M](http://vision.imar.ro/human3.6m/description.php).
After validating the transformer architecture on a toy limb a [SOTA](https://github.com/zhezh/adafuse-3d-human-pose) [pose estimation](https://github.com/facebookresearch/VideoPose3D) model will be used to extract poses from the videos and feed them to the transformer.

Potential use cases are modeling Movement Primitives in our VR lab or in the wild.

## Setup
Install with `mamba env create --file environment.yml` (tested with mamba and [mambaforge](https://github.com/conda-forge/miniforge#mambaforge), conda should work aswell albeit slow.)

Activate the environment with `mamba activate mp-transformer` and setup with `pip install -e .`.

## Run
Activate the environment and run training without logging to Weights & Biases with `python mp_transformer/train.py --no-log`.

## Demo
An example video of a reconstructed sequence can be found in `demo/example.mp4`.

### TODO:
- [x] improve toy example
- [x] overfit on toy data
- [ ] add translation to toy example
- [ ] Demo Jupyter-Notebook 
- [ ] hook up Adafuse or VideoPose3D and train end-to-end
- [ ] overfit on Walking subset of Human3.6M
- [ ] train properly on Human3.6M
- [ ] Eval/Demo