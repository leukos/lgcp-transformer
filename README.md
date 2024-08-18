This repository contains an all necessary code to reproduce the results of my experiments with permutation invariant and equivarint deep learning methods.

The code in models is practically a copy of https://github.com/rajesh-lab/deep_permutation_invariant.git, however I adapted the code to run in Pytorch DataParallel mode. The original source code was explicityly copying tensors to the first GPU device and was therfore not compatible with DataParallel.
