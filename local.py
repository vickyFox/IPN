from torchtools import *


class DefineDN4Net(nn.Module):
    def __init__(self, neighbor_k=3):
        super(DefineDN4Net, self).__init__()
        self.img_to_class = ImgToClassMetric(neighbor_k=neighbor_k)

    def forward(self, q, s1, s2):
        num_supports, c, h, w = s1.size()

        s1 = s1.view(num_supports, c, h, w)
        s2 = s2.view(num_supports, c, h, w)

        s = []
        support_set_sam = s1
        b, c, h, w = support_set_sam.size()
        support_set_sam = support_set_sam.permute(1, 0, 2, 3)
        support_set_sam = support_set_sam.contiguous().view(c, -1)
        s.append(support_set_sam)

        support_set_sam = s2
        b, c, h, w = support_set_sam.size()
        support_set_sam = support_set_sam.permute(1, 0, 2, 3)
        support_set_sam = support_set_sam.contiguous().view(c, -1)
        s.append(support_set_sam)

        x = self.img_to_class(q, s)  # get batch * num_classes

        return x


class ImgToClassMetric(nn.Module):
    def __init__(self, neighbor_k=3):
        super(ImgToClassMetric, self).__init__()
        self.neighbor_k = neighbor_k

    # Calculate the k-Nearest Neighbor of each local descriptor 
    def cal_cosine_similarity(self, input1, input2):
        b, c, h, w = input1.size()
        similarity_list = []

        for i in range(b):
            query_sam = input1[i]
            query_sam = query_sam.view(c, -1)
            query_sam = torch.transpose(query_sam, 0, 1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm

            if torch.cuda.is_available():
                inner_sim = torch.zeros(1, len(input2)).cuda()

            for j in range(len(input2)):
                support_set_sam = input2[j]
                support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
                support_set_sam = support_set_sam / support_set_sam_norm

                # cosine similarity between a query sample and a support category
                inner_product_matrix = query_sam @ support_set_sam

                # choose the top-k nearest neighbors
                top_k_value, top_k_index = torch.topk(inner_product_matrix, self.neighbor_k, 1)
                inner_sim[0, j] = torch.sum(top_k_value)

            similarity_list.append(inner_sim)

        similarity_list = torch.cat(similarity_list, 0)
        return similarity_list

    def forward(self, x1, x2):
        similarity_list = self.cal_cosine_similarity(x1, x2)
        return similarity_list


class DifNet(nn.Module):
    def __init__(self):
        super(DifNet, self).__init__()
        self.dn4_module = DefineDN4Net()

    def forward(self, feature):
        dn4_out_list = []
        for i in range(feature.size(0)):
            dn4_out = self.dn4_module(feature[i, 10:], feature[i, 0:5], feature[i, 5:10])
            dn4_out_list.append(dn4_out)
        dn4_out = torch.cat(dn4_out_list, 0)
        return dn4_out
