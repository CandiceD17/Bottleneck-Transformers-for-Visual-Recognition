# BoTNet-for-Visual-Recognition
### Introduction

PyTorch Implementation of **Bottleneck Transformers for Visual Recognition**. 

Link to paper: https://arxiv.org/abs/2101.11605

Structure of Self-Attention layer in paper:

![self-attention layer](https://github.com/CandiceD17/BoTNet-for-Visual-Recognition/blob/master/assets/self-attention-layer.png)

### Train the Network

To train BoTNet from scratch using the standard training process:

```shell
$ python3 train.py
```

To execute this code, you should have ImageNet downloaded in your local directory and correct path set in `IMAGE_DIR` in *train.py*.

You can also set the hyperparameters, including batch size, epoch, learning rate, and device inside *train.py*.

### References

Part of the codes are cited from:
https://github.com/lucidrains/bottleneck-transformer-pytorch
https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2