# DACNN release
This repository contains pytorch impelementation of the architectures proposed in the paper ["Deep Anchored Convolutional Neural Networks"](https://arxiv.org/abs/1904.09764).
(accepted to [CVPR2019 workshops](http://www.ee.oulu.fi/~lili/CEFRLatCVPR2019.html), oral)

All model files can be directly imported into pytorch training codes.

DACNN is a network stacked with a single convolution kernel across layers, 
incorperating 2 other weight sharing techniques coined "Mixed Architecture and Regulators".

<p align="center">
  <img src="../master/figures/sharing_techniques.png" width="800" alt="accessibility text">
</p>

As a network compression technique, the architecture achieved similar performances on CIFAR & SVHN dataset compared to 
some popular models (VGG, ResNet, etc.) while using much less parameters.

<p align="center">
  <img src="../master/figures/result_1.png" width="600" alt="accessibility text">
</p>
