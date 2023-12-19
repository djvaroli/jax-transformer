# Bumble Jax

<img src="./assets/images/logo.png" width="200" height="200" />

## The Name
Transformers are both alien robots and a model architecture (and other things). My favorite transformer was always Bumblebee so I decided to name this project accordingly. I explored other options, such as `optimusjx` and `jaximus-prime` but `bumblejax` proved itself to be the much more elegant candidate.

## Installation

### CUDA
Install dependencies:
```shell
pip install -e .
```
#### CUDA12
Install jax and jaxlib CUDA wheels
```shell
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### CUDA11
```shell
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

