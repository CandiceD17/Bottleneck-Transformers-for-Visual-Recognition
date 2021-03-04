# Bottleneck-Transformers-for-Visual-Recognition
## Introduction

PyTorch Implementation of **Bottleneck Transformers for Visual Recognition**. 

Link to paper: https://arxiv.org/abs/2101.11605

Structure of Self-Attention layer in paper:

![self-attention layer](https://github.com/CandiceD17/Bottleneck-Transformers-for-Visual-Recognition/blob/master/asset/self-attention-layer.png)

## Dependency

External libraries: eniops, yacs

Use pip command to install:

```shell
$ pip install eniops
$ pip install yacs
```

## Train the Network

To train BoTNet from scratch using the standard training process:

```shell
$ python3 train.py
```

To execute this code, you should have ImageNet downloaded in your local directory and correct path set in `_C.TRAIN.DATASET` and`_C.TEST.DATASET` in `util/config.py`.

You can use `.yaml` or pass in any parameters in command line to overwrite the configures in `config.py`.

For exmaple, to update `cfg.TRAIN.BATCH_SIZE`, you can either type `--train-batch_size` in command line or pass a `.yaml` file through `--cfg` using the below format:

```yaml
TRAIN:
  BATCH_SIZE: 256
```

## References

Part of the codes are cited from:

Bottleneck Transformer Backbone:

https://github.com/lucidrains/bottleneck-transformer-pytorch
https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2

Training Process:

https://github.com/BIGBALLON/distribuuuu (Biggest credit to my mentor at Sensetime!!)

## Features to come:

- Downsample at the first stage of attention layer with correct dimensions
- Object detection experiments for downstream task
