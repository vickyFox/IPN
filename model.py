from torchtools import *


class CCMNet(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features):
        super(CCMNet, self).__init__()
        self.num_layers = 1
        self.gru1 = nn.GRU(in_features * 5, hidden_features * 5, num_layers=self.num_layers, batch_first=True)
        self.gru2 = nn.GRU(in_features * 5, hidden_features * 5, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(in_features=in_features * 10, out_features=in_features, bias=True),
                                nn.ReLU(),
                                nn.Linear(in_features=in_features, out_features=in_features // 2, bias=True),
                                nn.ReLU(),
                                nn.Linear(in_features=in_features // 2, out_features=5, bias=True),
                                nn.Sigmoid())

    def forward(self, in_feat, num_supports):
        # in_feat: num_tasks(batch_size) x num_samples x in_features
        batch_size = in_feat.size(0)

        support_data = in_feat[:, :num_supports, :]
        output_list1 = []
        output_list2 = []
        for i in range(5):
            m_h_1 = in_feat[:, num_supports:, :].contiguous().view(batch_size, -1).unsqueeze(0).repeat(self.num_layers,
                                                                                                       1, 1)
            m_h_2 = m_h_1.clone()
            for j in range(5):
                output1, h_n1 = self.gru1(support_data[:, (5 * i) + j:(5 * i) + j + 1, :].repeat(1, 1, 5), m_h_1)
                output2, h_n2 = self.gru2(support_data[:, (5 * i) + 4 - j:(5 * i) + 5 - j, :].repeat(1, 1, 5), m_h_2)
                m_h_1 = (m_h_1 + h_n1) / 2
                m_h_2 = (m_h_2 + h_n2) / 2
                output_list1.append(output1)
                output_list2.append(output2)
            output_list2[5 * i: 5 * (i + 1)] = reversed(output_list2[5 * i: 5 * (i + 1)])

        output_data1 = torch.cat(output_list1, 1)
        output_data2 = torch.cat(output_list2, 1)
        output_data = torch.cat([output_data1, output_data2], -1)
        output_data = self.fc(output_data).transpose(1, 2)
        return output_data
