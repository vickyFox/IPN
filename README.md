# IPN
## Introduction
A pytorch implementation of the IJCAI2020 paper "[Few-shot Visual Learning with Contextual Memory and Fine-grained Calibration](http://static.ijcai.org/2020-accepted_papers.html)". The code is based on [Edge-labeling Graph Neural Network for Few-shot Learning](https://github.com/khy0809/fewshot-egnn) and [Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning](https://github.com/WenbinLee/DN4).

**Author:** Yuqing Ma, Wei Liu, Shihao Bai, Qingyu Zhang, Aishan Liu, Weimin Chen and Xianglong Liu

**Abstract:** Few-shot learning aims to learn a model that can be readily adapted to new unseen classes (concepts) by accessing one or few examples. Despite the successful progress, most of the few-shot learning approaches, concentrating on either global or local characteristics of examples, still suffer from weak generalization abilities. Inspired by the inverted pyramid theory, to address this problem, we propose an inverted pyramid network (IPN) that intimates the human's coarse-to-fine cognition paradigm. The proposed IPN consists of two consecutive stages, namely global stage and local stage. At the global stage, a class-sensitive contextual memory network (CCMNet) is introduced to learn discriminative support-query relation embeddings and predict the query-to-class similarity based on the contextual memory. Then at the local stage, a fine-grained calibration is further appended to complement the coarse relation embeddings, targeting more precise query-to-class similarity evaluation. To the best of our knowledge, IPN is the first work that simultaneously integrates both global and local characteristics in few-shot learning, approximately imitating the human cognition mechanism. Our extensive experiments on multiple benchmark datasets demonstrate the superiority of IPN, compared to a number of state-of-the-art approaches.

## Requirements
* Python 3
* Python packages
  - pytorch 1.0.0
  - torchvision 0.2.2
  - matplotlib
  - numpy
  - pillow
  - tensorboardX

An NVIDIA GPU and CUDA 9.0 or higher. 

## Getting started
### mini-ImageNet
You can download miniImagenet dataset from [here](https://drive.google.com/drive/folders/15WuREBvhEbSWo4fTr1r-vMY0C_6QWv4w).

### tiered-ImageNet
You can download tieredImagenet dataset from [here](https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view?usp=drive_open).


Because WRN has a large amount of parameters. You can save the extracted feature before the classifaction layer to increase train or test speed. Here we provide the features extracted by WRN:
* miniImageNet: [train](https://drive.google.com/file/d/1uJ5-NhdDkdkqRhyrQoXKgkqoLt3BqWSC/view?usp=sharing), [val](https://drive.google.com/file/d/1p_6kalUR-a2so1yOGUn1DCAXL3ftgl-r/view?usp=sharing), [test](https://drive.google.com/file/d/1z69BN3ReZfSwpOt3P1l1LPDdqigKdsfT/view?usp=sharing)
* tieredImageNet: [train](https://drive.google.com/file/d/1dGtfL8EEplJmiXGgxmQNtI36FYKyp-XG/view?usp=sharing), [val](https://drive.google.com/file/d/1DQ-LsyWtFsi6oyTxnBa5nQrla6lY7x0M/view?usp=sharing), [test](https://drive.google.com/file/d/1dGtfL8EEplJmiXGgxmQNtI36FYKyp-XG/view?usp=sharing)

You also can use our [pretrained WRN model](https://drive.google.com/drive/folders/1o51s2F7_bpG2k6JOgE9loYtSRIdOH2qc) to generate features for mini or tiered by yourself.

## Training
```
# ************************** miniImagenet, 5way 1shot  *****************************
$ python3 train.py --dataset mini --num_ways 5 --num_shots 1 

# ************************** miniImagenet, 5way 5shot *****************************
$ python3 train.py --dataset mini --num_ways 5 --num_shots 5 

# ************************** tieredImagenet, 5way 1shot *****************************
$ python3 train.py --dataset tiered --num_ways 5 --num_shots 1 

# ************************** tieredImagenet, 5way 5shot *****************************
$ python3 train.py --dataset tiered --num_ways 5 --num_shots 5 

```
You can download our pretrained model from [here](https://drive.google.com/drive/folders/1dsjQNAAcxa8e2WIHRYi5WxRiT2tqJkbR?usp=sharing) to reproduce the results of the paper.
## Testing
``` 
# ************************** miniImagenet, 5way 5shot *****************************
$ python3 eval.py --test_model your_path --dataset mini --num_ways C --num_shots K 

```
