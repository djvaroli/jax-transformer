# Jax Transformer (wheel-jax)

<img src="./assets/images/logo.png" width="200" height="200" />

## Why Wheel Jax?
Transformers are both alien robots and a model architecture (and other things too). Quite a few of the jax libraries use the "ax" sound in one form or another so I decided that a good name for this project would be some combination of something from "The Transformers" and  the "ax" sound. I tried a few ideas, "optimus-jax" which was hard to type out, "bumble jax" which wasn't a great name. Finally I found a character from "The Transformer" franchise that had the perfect name: "Wheeljack". WheelJax seemed like a natural fit.

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

