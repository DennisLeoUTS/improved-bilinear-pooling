# improved-bilinear-pooling
Pytorch implementation of BCNN and iBCNN.

Introduction
This repo contains codes for fine tuning BCNN and iBCNN on CUB_200_2011 datasets.

Ref. to:

[1]. Lin, Tsung-Yu, Aruni RoyChowdhury, and Subhransu Maji. "Bilinear cnn models for fine-grained visual recognition." Proceedings of the IEEE international conference on computer vision. 2015.
[2]. Lin, Tsung-Yu, and Subhransu Maji. "Improved bilinear pooling with cnns." arXiv preprint arXiv:1707.06772 (2017).

Datasets
1.CUB200-2011
    CUB-200-2011 dataset has 11,788 images of 200 bird species.

How to use
git clone https://github.com/DennisLeoUTS/improved-bilinear-pooling.git
cd improved-bilinear-pooling

modify the data_path in utils/Config.py. The path should include data package CUB_200_2011.tgz

To finetune the bilinear CNN in [1]:
    python trainer_bilinear_ft_last_layer.py
after finished, run:
    python trainer_bilinear_ft_all.py
and it can give 84.98% top-1 accuracy.

To finetune the improved bilinear CNN in [2]:
    python trainer_improved_bilinear_ft_last_layer.py
after finished, run:
    python trainer_improved_bilinear_ft_all.py
and it can give 85.94% top-1 accuracy.
