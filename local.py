from torchtools import *
from collections import OrderedDict
import math
import numpy as np
import matplotlib.pyplot as plt

class define_DN4Net(nn.Module):
    def __init__(self, num_classes=5, neighbor_k=3):
        super(define_DN4Net, self).__init__()

    
        self.imgtoclass = ImgtoClass_Metric(neighbor_k=neighbor_k)  


    def forward(self, q, s1, s2):
        # q: 1 x 256 x 5 x 5
        # s1: 5 x 256 x 5 x 5
        # s2: 5 x 256 x 5 x 5

        num_supports, C, h, w = s1.size()
        
        s1 = s1.view(num_supports, C, h, w)
        s2 = s2.view(num_supports, C, h, w)
        
        S = []
        support_set_sam = s1
        B, C, h, w = support_set_sam.size()
        support_set_sam = support_set_sam.permute(1, 0, 2, 3)
        support_set_sam = support_set_sam.contiguous().view(C, -1)
        S.append(support_set_sam)
        
        support_set_sam = s2
        B, C, h, w = support_set_sam.size()
        support_set_sam = support_set_sam.permute(1, 0, 2, 3)
        support_set_sam = support_set_sam.contiguous().view(C, -1)
        S.append(support_set_sam)

        x = self.imgtoclass(q, S) # get Batch*num_classes

        return x

#========================== Define an image-to-class layer ==========================#


class ImgtoClass_Metric(nn.Module):
    def __init__(self, neighbor_k=3):
        super(ImgtoClass_Metric, self).__init__()
        self.neighbor_k = neighbor_k


    # Calculate the k-Nearest Neighbor of each local descriptor 
    def cal_cosinesimilarity(self, input1, input2):
        B, C, h, w = input1.size()
        Similarity_list = []

        for i in range(B):
            query_sam = input1[i]
            query_sam = query_sam.view(C, -1)
            query_sam = torch.transpose(query_sam, 0, 1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)   
            query_sam = query_sam/query_sam_norm

            if torch.cuda.is_available():
                inner_sim = torch.zeros(1, len(input2)).cuda()

            for j in range(len(input2)):
                support_set_sam = input2[j]
                support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
                support_set_sam = support_set_sam/support_set_sam_norm

                # cosine similarity between a query sample and a support category
                innerproduct_matrix = query_sam@support_set_sam

                # choose the top-k nearest neighbors
                topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)
                inner_sim[0, j] = torch.sum(topk_value)

            Similarity_list.append(inner_sim)

        Similarity_list = torch.cat(Similarity_list, 0)    

        return Similarity_list 


    def forward(self, x1, x2):

        Similarity_list = self.cal_cosinesimilarity(x1, x2)

        return Similarity_list






class DifferNet(nn.Module):
    def __init__(self):
        super(DifferNet, self).__init__()
        self.dn4_module = define_DN4Net()
        
    def forward(self, input_data, feature_4):
        # input_data: batch_size x 11 x 3 x 84 x 84
        # query_data: batch_size x 1 x 256 x 5 x 5

        score_list = []
        output_data_3d = feature_4
        dn4_out_list = []

        for i in range(input_data.size(0)):
            dn4_out = self.dn4_module(output_data_3d[i, 10:], output_data_3d[i, 0:5], output_data_3d[i, 5:10])
            dn4_out_list.append(dn4_out)
        dn4_out_list = torch.cat(dn4_out_list, 0)

        return dn4_out_list




