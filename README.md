# TRPN
## Introduction
A pytorch implementation of the IJCAI2020 paper "[Few-shot Visual Learning with Contextual Memory and Fine-grained Calibration](http://static.ijcai.org/2020-accepted_papers.html)". The code is based on [Edge-labeling Graph Neural Network for Few-shot Learning](https://github.com/khy0809/fewshot-egnn)

**Author:** Yuqing Ma, Wei Liu, Shihao Bai, Qingyu Zhang, Aishan Liu, Weimin Chen and Xianglong Liu

**Abstract:** Few-shot learning aims to learn a model that can be readily adapted to new unseen classes (concepts) by accessing one or few examples. Despite the successful progress, most of the few-shot learning approaches, concentrating on either global or local characteristics of examples, still suffer from weak generalization abilities. Inspired by the inverted pyramid theory, to address this problem, we propose an inverted pyramid network (IPN) that intimates the human's coarse-to-fine cognition paradigm. The proposed IPN consists of two consecutive stages, namely global stage and local stage. At the global stage, a class-sensitive contextual memory network (CCMNet) is introduced to learn discriminative support-query relation embeddings and predict the query-to-class similarity based on the contextual memory. Then at the local stage, a fine-grained calibration is further appended to complement the coarse relation embeddings, targeting more precise query-to-class similarity evaluation. To the best of our knowledge, IPN is the first work that simultaneously integrates both global and local characteristics in few-shot learning, approximately imitating the human cognition mechanism. Our extensive experiments on multiple benchmark datasets demonstrate the superiority of IPN, compared to a number of state-of-the-art approaches.
