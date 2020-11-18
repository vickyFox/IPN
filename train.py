from torchtools import *
from data import MiniImagenetLoader, TieredImagenetLoader
import shutil
import os
import random
from model import *
from local import *
class ModelTrainer(object):
    def __init__(self,
                 enc_module,
                 ccmNet_module,
                 dif_module,
                 data_loader):
        self.enc_module = enc_module.to(tt.arg.device)
        self.ccmNet_module = ccmNet_module.to(tt.arg.device)
        self.dif_module = dif_module.to(tt.arg.device)
        if tt.arg.num_gpus > 1:
            print('Construct multi-gpu model ...')
            self.enc_module = nn.DataParallel(self.enc_module, device_ids=[0, 1], dim=0)
            self.ccmNet_module = nn.DataParallel(self.ccmNet_module, device_ids=[0, 1], dim=0)
            self.dif_module = nn.DataParallel(self.dif_module, device_ids=[0, 1], dim=0)

            print('done!\n')

        # get data loader
        self.data_loader = data_loader

        # set optimizer
        self.module_params = list(self.enc_module.parameters()) + list(self.ccmNet_module.parameters())

        # set optimizer
        self.optimizer = optim.Adam(params=self.module_params,
                                    lr=tt.arg.lr,
                                    weight_decay=tt.arg.weight_decay)

        # set loss
        self.bce_loss = nn.BCELoss()

        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0

    def train(self):
        val_acc = self.val_acc

        # set edge mask (to distinguish support and query edges)
        num_supports = tt.arg.num_ways_train * tt.arg.num_shots_train
        num_queries = tt.arg.num_ways_train * 1
        num_samples = num_supports + num_queries
        support_edge_mask = torch.zeros(tt.arg.meta_batch_size, num_samples, num_samples).to(tt.arg.device)
        support_edge_mask[:, :num_supports, :num_supports] = 1
        query_edge_mask = 1 - support_edge_mask

        # for each iteration
        for iter in range(self.global_step + 1, tt.arg.train_iteration + 1):
            # init grad
            self.optimizer.zero_grad()

            # set current step
            self.global_step = iter

            # load task data list
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader['train'].get_task_batch(num_tasks=tt.arg.meta_batch_size,
                                                                     num_ways=tt.arg.num_ways_train,
                                                                     num_shots=tt.arg.num_shots_train,
                                                                     seed=iter + tt.arg.seed)

            # set as single data
            full_data = torch.cat([support_data, query_data], 1)
            full_label = torch.cat([support_label, query_label], 1)

            # 包含了所有边的特征
            # batch_size x 2 x num_samples x num_samples
            full_edge = self.label2edge(full_label)

            # set init edge
            # batch_size x 2 x num_samples x num_samples
            init_edge = full_edge.clone()
            init_edge[:, :, num_supports:, :] = 0.5
            init_edge[:, :, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
                init_edge[:, 1, num_supports + i, num_supports + i] = 0.0
                
            # set as train mode
            self.enc_module.train()
            self.ccmNet_module.train()

            # (1) encode data
            full_data = [self.enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1) # batch_size x num_samples x featdim


            query_score_list = self.ccmNet_module(in_feat=full_data, num_supports=num_supports)
            query_score_list = query_score_list.view(tt.arg.meta_batch_size, num_queries, num_supports)

            # (4) compute loss
            loss2 = self.bce_loss(query_score_list, full_edge[:, 0, num_supports:, :num_supports]).mean()

            # compute node accuracy: num_tasks x num_queries x num_ways == {num_tasks x num_queries x num_supports} * {num_tasks x num_supports x num_ways}
           
            query_node_pred4 = torch.bmm(query_score_list, self.one_hot_encode(tt.arg.num_ways_train, support_label.long()))
            query_node_accr4 = torch.eq(torch.max(query_node_pred4, -1)[1], query_label.long()).float().mean()
            
            
            total_loss = loss2
            total_loss.backward()
#             print(total_loss)
            self.optimizer.step()

            # adjust learning rate
            self.adjust_learning_rate(optimizers=[self.optimizer],
                                      lr=tt.arg.lr,
                                      iter=self.global_step)

            # logging
            tt.log_scalar('train/edge_loss', total_loss, self.global_step)
            tt.log_scalar('train/node_accr4_final', query_node_accr4, self.global_step)

            # evaluation
            if self.global_step % tt.arg.test_interval == 0:
                val_acc = self.eval(partition='val')

                is_best = 0

                if val_acc >= self.val_acc:
                    self.val_acc = val_acc
                    is_best = 1

                tt.log_scalar('val/best_accr', self.val_acc, self.global_step)

                self.save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'ccmNet_module_state_dict': self.ccmNet_module.state_dict(),
                    'val_acc': val_acc,
                    'optimizer': self.optimizer.state_dict(),
                    }, is_best)

            tt.log_step(global_step=self.global_step)

    def eval(self, partition='test', log_flag=True):
        best_acc = 0
        # set edge mask (to distinguish support and query edges)
        num_supports = tt.arg.num_ways_test * tt.arg.num_shots_test
        num_queries = tt.arg.num_ways_test * 1
        num_samples = num_supports + num_queries
        support_edge_mask = torch.zeros(tt.arg.test_batch_size, num_samples, num_samples).to(tt.arg.device)
        support_edge_mask[:, :num_supports, :num_supports] = 1
        query_edge_mask = 1 - support_edge_mask
        
        query_edge_losses = []
        query_node_accrs4 = []
        query_node_accrs5 = []
        f = open("out.txt", "w")
        # for each iteration
        for iter in range(tt.arg.test_iteration//tt.arg.test_batch_size):
            # load task data list
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader[partition].get_task_batch(num_tasks=tt.arg.test_batch_size,
                                                                       num_ways=tt.arg.num_ways_test,
                                                                       num_shots=tt.arg.num_shots_test,
                                                                       seed=iter)

            # set as single data
            full_data_init = torch.cat([support_data, query_data], 1)
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = self.label2edge(full_label)

            # set init edge
            init_edge = full_edge.clone()
            init_edge[:, :, num_supports:, :] = 0.5
            init_edge[:, :, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
                init_edge[:, 1, num_supports + i, num_supports + i] = 0.0

            # set as train mode
            self.enc_module.eval()
            self.ccmNet_module.eval()
            self.dif_module.eval()
            # (1) encode data
           


           # (1) encode data
            full_data_list = []
            full_data_list_4 = []
            for data in full_data_init.chunk(full_data_init.size(1), dim=1):
                
                output_data_4, output_data = self.enc_module(data.squeeze(1))
                full_data_list_4.append(output_data_4)
                full_data_list.append(output_data)
                
            full_data = torch.stack(full_data_list, dim=1) # batch_size x num_samples x featdim
            full_data_4 = torch.stack(full_data_list_4, dim=1) # batch_size x num_samples x c x h x w
            #print(full_data.shape)
            query_score_list = self.ccmNet_module(in_feat=full_data, num_supports=num_supports)

            query_score_list = query_score_list.view(tt.arg.test_batch_size, num_queries, num_supports)


            loss2 = self.bce_loss(query_score_list, full_edge[:, 0, num_supports:, :num_supports]).mean()

            # compute node accuracy: num_tasks x num_queries x num_ways == {num_tasks x num_queries x num_supports} * {num_tasks x num_supports x num_ways}
           
            query_node_pred4 = torch.bmm(query_score_list, self.one_hot_encode(tt.arg.num_ways_test, support_label.long()))
            query_node_accr4 = torch.eq(torch.max(query_node_pred4, -1)[1], query_label.long()).float().mean()
            query_node_accr5 = torch.eq(torch.max(query_node_pred4, -1)[1], query_label.long()).float()
           


            num_supports = tt.arg.num_ways_test * tt.arg.num_shots_test
            num_queries = tt.arg.num_ways_test * 1
            num_samples = num_supports + num_queries
            batch_size = tt.arg.test_batch_size
            num_ways = tt.arg.num_ways_test
            num_shots = tt.arg.num_shots_test

            support_label_tiled = support_label.unsqueeze(1).repeat(1, num_queries, 1).view(tt.arg.test_batch_size * num_queries, num_supports)
            query_label_reshaped = query_label.contiguous().view(tt.arg.test_batch_size * num_queries, 1)


            full_label_reshaped = torch.cat([support_label_tiled, query_label_reshaped], 1)
            gcnfg_input_list = []
            gcnfg_input_list_3 = []

            gcnfg_input_list_4 = []
            global_score = []
            global_pred = []
            sel = []
            gcnfg_label_list = []
            ij_list = []
            values, indices = query_score_list[:, :, :num_supports].view(tt.arg.test_batch_size, num_queries, num_ways, num_shots).mean(-1).sort(dim=-1, descending=True)
            flag = False
            for i in range(batch_size):
                is_global = False
                for j in range(num_queries):
                    if values[i, j, 1] > 0 and values[i, j, 0] / values[i, j, 1] < 1.5: # reject
                        flag = True
                        input_tmp = torch.cat((full_data_init[i, (5*indices[i,j,0]):(5*indices[i,j,0]+5), :], full_data_init[i, (5*indices[i,j,1]):(5*indices[i,j,1]+5), :], full_data_init[i, num_supports+j, :].unsqueeze(0)), 0)
                        gcnfg_input_list.append(input_tmp)
                        sel.append((indices[i,j,0], indices[i,j,1]))
                        global_score.append(query_node_pred4[i][j])

                        input_tmp_4 = torch.cat((full_data_4[i, (5*indices[i,j,0]):(5*indices[i,j,0]+5)], full_data_4[i, (5*indices[i,j,1]):(5*indices[i,j,1]+5)], full_data_4[i, num_supports+j].unsqueeze(0)), 0)
                        gcnfg_input_list_4.append(input_tmp_4)

                        label_tmp = torch.cat((full_label_reshaped[i*num_queries+j, (5*indices[i,j,0]):(5*indices[i,j,0]+5)], full_label_reshaped[i*num_queries+j, (5*indices[i,j,1]):(5*indices[i,j,1]+5)], full_label_reshaped[i*num_queries+j, num_supports:]), 0)
                        gcnfg_label_list.append(label_tmp)
                        is_global = True
                        ij_list.append((i,j))



            if flag:
                gcnfg_input = torch.stack(gcnfg_input_list, 0)
                gcnfg_input_4 = torch.stack(gcnfg_input_list_4, 0)

                gcnfg_label = torch.stack(gcnfg_label_list, 0)
                gcnfg_edge_label = self.label2edge_2(gcnfg_label)

                dn4_out_list = self.dif_module(input_data=gcnfg_input, feature_4=gcnfg_input_4)

                # dif_loss = self.bce_loss(score_list, torch.zeros(score_list.size()).to(tt.arg.device))

                # dn4_out_tmp = torch.zeros(dn4_out_list.size(0), 5).to(tt.arg.device)
                for k in range(dn4_out_list.size(0)):
                    i,j = ij_list[k]
                    query_node_accr5[i, j] = (indices[i, j, torch.max(dn4_out_list[k], -1)[1]] == query_label[i, j].long())
            query_node_accr5 = query_node_accr5.mean()
 
            total_losy = loss2
#             print(total_loss)
            query_node_accrs4 += [query_node_accr4.item()]
            query_node_accrs5 += [query_node_accr5.item()]
            

        # logging
        if log_flag:
            tt.log('---------------------------')
    
            tt.log_scalar('{}/node_accr4_final'.format(partition), np.array(query_node_accrs4).mean(), self.global_step)
            tt.log_scalar('{}/node_accr5_final'.format(partition), np.array(query_node_accrs5).mean(), self.global_step)

           
            tt.log('evaluation: total_count=%d, accuracy4_ccmnet: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                   (iter,
                    np.array(query_node_accrs4).mean() * 100,
                    np.array(query_node_accrs4).std() * 100,
                    1.96 * np.array(query_node_accrs4).std() / np.sqrt(float(len(np.array(query_node_accrs4)))) * 100))
            tt.log('evaluation: total_count=%d, accuracy4_TPN: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                   (iter,
                    np.array(query_node_accrs5).mean() * 100,
                    np.array(query_node_accrs5).std() * 100,
                    1.96 * np.array(query_node_accrs5).std() / np.sqrt(float(len(np.array(query_node_accrs5)))) * 100))
            tt.log('---------------------------')

        return np.array(query_node_accrs5).mean()



    def label2edge_2(self, label):
        # get size
        num_samples = label.size(1)

        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)

        # compute edge
        edge = torch.eq(label_i, label_j).float().to(tt.arg.device)

        return edge





    def adjust_learning_rate(self, optimizers, lr, iter):
        new_lr = lr * (0.5 ** (int(iter / tt.arg.dec_lr)))

        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    def label2edge(self, label):
        # get size
        num_samples = label.size(1)

        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)

        # compute edge
        edge = torch.eq(label_i, label_j).float().to(tt.arg.device)

        # expand
        edge = edge.unsqueeze(1)
        edge = torch.cat([edge, 1 - edge], 1)
        return edge

    def hit(self, logit, label):
        pred = logit.max(1)[1]
        hit = torch.eq(pred, label).float()
        return hit

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes)[class_idx].to(tt.arg.device)

    def save_checkpoint(self, state, is_best):
        torch.save(state, 'asset/checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar')
        if is_best:
            shutil.copyfile('asset/checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar',
                            'asset/checkpoints/{}/'.format(tt.arg.experiment) + 'model_best.pth.tar')



if __name__ == '__main__':

    tt.arg.device = 'cuda:0' if tt.arg.device is None else tt.arg.device
    # replace dataset_root with your own
    tt.arg.dataset_root = '/root/mayuqing/'
    tt.arg.dataset = 'mini' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 5 if tt.arg.num_shots is None else tt.arg.num_shots
    tt.arg.meta_batch_size = 20 if tt.arg.meta_batch_size is None else tt.arg.meta_batch_size
    tt.arg.transductive = False if tt.arg.transductive is None else tt.arg.transductive
    tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_gpus = 1 if tt.arg.num_gpus is None else tt.arg.num_gpus

    tt.arg.num_ways_train = tt.arg.num_ways
    tt.arg.num_ways_test = tt.arg.num_ways

    tt.arg.num_shots_train = tt.arg.num_shots
    tt.arg.num_shots_test = tt.arg.num_shots

    tt.arg.train_transductive = tt.arg.transductive
    tt.arg.test_transductive = tt.arg.transductive

    # model parameter related
    tt.arg.emb_size = 640

    # train, test parameters
    tt.arg.train_iteration = 100000 if tt.arg.dataset == 'mini' else 200000
    tt.arg.test_iteration = 10000
    tt.arg.test_interval = 5000 if tt.arg.test_interval is None else tt.arg.test_interval
    tt.arg.test_batch_size = 10
    tt.arg.log_step = 100 if tt.arg.log_step is None else tt.arg.log_step

    tt.arg.lr = 1e-3
    tt.arg.grad_clip = 5
    tt.arg.weight_decay = 1e-6
    tt.arg.dec_lr = 15000 if tt.arg.dataset == 'mini' else 30000
    tt.arg.dropout = 0.1 if tt.arg.dataset == 'mini' else 0.0

    tt.arg.experiment = 'model' if tt.arg.experiment is None else tt.arg.experiment

    #set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tt.arg.log_dir_user = tt.arg.log_dir if tt.arg.log_dir_user is None else tt.arg.log_dir_user
    tt.arg.log_dir = tt.arg.log_dir_user

    if not os.path.exists('asset/checkpoints'):
        os.makedirs('asset/checkpoints')
    if not os.path.exists('asset/checkpoints/' + tt.arg.experiment):
        os.makedirs('asset/checkpoints/' + tt.arg.experiment)


    enc_module = wideres(num_classes = 64)

    ccmNet_module = CCMNet(in_features=tt.arg.emb_size, hidden_features=tt.arg.emb_size)
    dif_module = DifferNet()

    

    if tt.arg.dataset == 'mini':
        train_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='train')
        valid_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='val')
    elif tt.arg.dataset == 'tiered':
        train_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='train')
        valid_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='val')
    else:
        print('Unknown dataset!')

    data_loader = {'train': train_loader,
                   'val': valid_loader
                   }

    # create trainer
    trainer = ModelTrainer(enc_module=enc_module,
                           ccmNet_module=ccmNet_module,
                           dif_module = dif_module,
                           data_loader=data_loader)

    trainer.train()
