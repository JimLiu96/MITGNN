# Introduction
Repo for paper [Basket Recommendation with Multi-Intent Translation Graph Neural Network](https://arxiv.org/abs/2010.11419)

# Citation
If you use the code, please cite our paper
```
@INPROCEEDINGS{9006266,
  author={Z. {Liu} and X. {Li} and Z. {Fan} and S. {Guo} and K. {Achan}} and P. S. {Yu}},
  booktitle={2020 IEEE International Conference on Big Data (Big Data)},
  title={Basket Recommendation with Multi-Intent Translation Graph Neural Network},
  year={2020}
```

# Usage
```
python MITGNN.py --dataset inscart_3 --regs [1e-3] --alg_type intent_conv --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 4096 --num_intent 5 --epoch 1000
```
## Environment
Python = 3.6
Tensorflow = 1.8+
Numpy, Scipy, scikit-learn should be installed accordingly.

# Acknoledgement
We reuse some code from our previous paper [basConv](https://github.com/JimLiu96/basConv). You may refer this code for more information on the dataset. 
