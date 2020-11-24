import shutil
import os
import random

from torchtools import *
from data import MiniImagenetLoader, TieredImagenetLoader
from backbone.WideResNet import wide_res
from model import CCMNet
from local import DifNet


class ModelTrainer(object):
    def __init__(self,
                 enc_module,
                 ccmnet_module,
                 dif_module,
                 data_loader):
        self.enc_module = enc_module.to(tt.arg.device)
        self.ccmnet_module = ccmnet_module.to(tt.arg.device)
        self.dif_module = dif_module.to(tt.arg.device)
        if tt.arg.num_gpus > 1:
            print('Construct multi-gpu model ...')
            self.enc_module = nn.DataParallel(self.enc_module, device_ids=[0, 1], dim=0)
            self.ccmnet_module = nn.DataParallel(self.ccmnet_module, device_ids=[0, 1], dim=0)
            self.dif_module = nn.DataParallel(self.dif_module, device_ids=[0, 1], dim=0)

            print('done!\n')

        # get data loader
        self.data_loader = data_loader

        # set optimizer
        self.module_params = list(self.enc_module.parameters()) + list(self.ccmnet_module.parameters())
        self.optimizer = optim.Adam(params=self.module_params,
                                    lr=tt.arg.lr,
                                    weight_decay=tt.arg.weight_decay)

        # set loss
        self.bce_loss = nn.BCELoss()

        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0

    def train(self):
        num_supports = tt.arg.num_ways_train * tt.arg.num_shots_train
        num_queries = tt.arg.num_ways_train * 1

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

            # batch_size x 2 x num_samples x num_samples
            full_edge = self.label2edge(full_label)

            # set as train mode
            self.enc_module.train()
            self.ccmnet_module.train()

            # encode data
            full_data = [self.enc_module(data.squeeze(1))[1] for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1)  # batch_size x num_samples x feat_dim

            query_score_matrix = self.ccmnet_module(in_feat=full_data, num_supports=num_supports)
            query_score_matrix = query_score_matrix.view(tt.arg.meta_batch_size, num_queries, num_supports)

            # compute loss
            loss = self.bce_loss(query_score_matrix, full_edge[:, 0, num_supports:, :num_supports]).mean()

            # compute node accuracy: num_tasks x num_queries x num_ways ==
            # {num_tasks x num_queries x num_supports} * {num_tasks x num_supports x num_ways}
            query_pred_ccmnet = torch.bmm(query_score_matrix,
                                          self.one_hot_encode(tt.arg.num_ways_train, support_label.long()))
            query_acc_ccmnet = torch.eq(torch.max(query_pred_ccmnet, -1)[1], query_label.long()).float().mean()

            loss.backward()
            self.optimizer.step()

            # adjust learning rate
            self.adjust_learning_rate(optimizers=[self.optimizer],
                                      lr=tt.arg.lr,
                                      iter=self.global_step)

            # logging
            tt.log_scalar('train/loss', loss, self.global_step)
            tt.log_scalar('train/query_acc_ccmnet', query_acc_ccmnet, self.global_step)

            # evaluation
            if self.global_step % tt.arg.test_interval == 0:
                val_acc = self.eval(partition='val')
                is_best = 0
                if val_acc >= self.val_acc:
                    self.val_acc = val_acc
                    is_best = 1

                tt.log_scalar('val/best_acc', self.val_acc, self.global_step)

                self.save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'ccmnet_module_state_dict': self.ccmnet_module.state_dict(),
                    'val_acc': val_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best)

            tt.log_step(global_step=self.global_step)

    def eval(self, partition='test', log_flag=True):
        batch_size = tt.arg.test_batch_size
        num_supports = tt.arg.num_ways_test * tt.arg.num_shots_test
        num_queries = tt.arg.num_ways_test * 1

        query_acc_list_ccmnet = []
        query_acc_list_ipn = []

        # for each iteration
        for iter in range(tt.arg.test_iteration // tt.arg.test_batch_size):
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

            # set as train mode
            self.enc_module.eval()
            self.ccmnet_module.eval()
            self.dif_module.eval()

            # (1) encode data
            full_global_feat_list = []
            full_local_feat_list = []
            for data in full_data_init.chunk(full_data_init.size(1), dim=1):
                local_feat, global_feat = self.enc_module(data.squeeze(1))
                full_local_feat_list.append(local_feat)
                full_global_feat_list.append(global_feat)

            full_global_feat = torch.stack(full_global_feat_list, dim=1)  # batch_size x num_samples x feat_dim
            full_local_feat = torch.stack(full_local_feat_list, dim=1)  # batch_size x num_samples x c x h x w

            query_score_matrix = self.ccmnet_module(in_feat=full_global_feat, num_supports=num_supports)
            query_score_matrix = query_score_matrix.view(tt.arg.test_batch_size, num_queries, num_supports)

            # compute node accuracy: num_tasks x num_queries x num_ways ==
            # {num_tasks x num_queries x num_supports} * {num_tasks x num_supports x num_ways}
            query_pred_ccmnet = torch.bmm(query_score_matrix,
                                          self.one_hot_encode(tt.arg.num_ways_test, support_label.long()))
            query_acc_ccmnet = torch.eq(torch.max(query_pred_ccmnet, -1)[1], query_label.long()).float().mean()
            query_acc_list_ccmnet += [query_acc_ccmnet.item()]

            # init query_acc_ipn
            query_acc_ipn = torch.eq(torch.max(query_pred_ccmnet, -1)[1], query_label.long()).float()

            dif_input_feature_list = []
            dif_input_index_list = []
            query_pred_sorted, query_index = query_pred_ccmnet.sort(dim=-1, descending=True)
            dif_flag = False
            for i in range(batch_size):
                for j in range(num_queries):
                    if query_pred_sorted[i, j, 1] > 0 and query_pred_sorted[i, j, 0] / query_pred_sorted[i, j, 1] < 1.5:
                        dif_flag = True
                        dif_input_index_list.append((i, j))
                        dif_input_feature = torch.cat(
                            (full_local_feat[i, (5 * query_index[i, j, 0]):(5 * query_index[i, j, 0] + 5)],
                             full_local_feat[i, (5 * query_index[i, j, 1]):(5 * query_index[i, j, 1] + 5)],
                             full_local_feat[i, num_supports + j].unsqueeze(0)), 0)
                        dif_input_feature_list.append(dif_input_feature)

            if dif_flag:
                dif_input_features = torch.stack(dif_input_feature_list, 0)
                dif_output = self.dif_module(feature=dif_input_features)
                for k in range(dif_output.size(0)):
                    i, j = dif_input_index_list[k]
                    query_acc_ipn[i, j] = (
                            query_index[i, j, torch.max(dif_output[k], -1)[1]] == query_label[i, j].long())

            query_acc_ipn = query_acc_ipn.mean()
            query_acc_list_ipn += [query_acc_ipn.item()]

        # logging
        if log_flag:
            tt.log('---------------------------')

            tt.log_scalar('{}/query_acc_ccmnet'.format(partition), np.array(query_acc_list_ccmnet).mean(),
                          self.global_step)
            tt.log_scalar('{}/query_acc_ipn'.format(partition), np.array(query_acc_list_ipn).mean(),
                          self.global_step)

            tt.log('evaluation: total_count=%d, accuracy_CCMNet: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                   (iter,
                    np.array(query_acc_list_ccmnet).mean() * 100,
                    np.array(query_acc_list_ccmnet).std() * 100,
                    1.96 * np.array(query_acc_list_ccmnet).std() / np.sqrt(
                        float(len(np.array(query_acc_list_ccmnet)))) * 100))
            tt.log('evaluation: total_count=%d, accuracy_TPN: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                   (iter,
                    np.array(query_acc_list_ipn).mean() * 100,
                    np.array(query_acc_list_ipn).std() * 100,
                    1.96 * np.array(query_acc_list_ipn).std() / np.sqrt(
                        float(len(np.array(query_acc_list_ipn)))) * 100))
            tt.log('---------------------------')

        return np.array(query_acc_list_ipn).mean()

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
    tt.arg.dataset_root = '/root/IPN/'
    tt.arg.dataset = 'mini' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 5 if tt.arg.num_shots is None else tt.arg.num_shots
    tt.arg.meta_batch_size = 20 if tt.arg.meta_batch_size is None else tt.arg.meta_batch_size
    tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_gpus = 1 if tt.arg.num_gpus is None else tt.arg.num_gpus

    tt.arg.num_ways_train = tt.arg.num_ways
    tt.arg.num_ways_test = tt.arg.num_ways

    tt.arg.num_shots_train = tt.arg.num_shots
    tt.arg.num_shots_test = tt.arg.num_shots

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

    tt.arg.experiment = 'WRN_mini_5_5' if tt.arg.experiment is None else tt.arg.experiment

    # set random seed
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

    enc_module = wide_res(num_classes=64, remove_linear=True)
    ccmnet_module = CCMNet(in_features=tt.arg.emb_size, hidden_features=tt.arg.emb_size)
    dif_module = DifNet()

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
                           ccmnet_module=ccmnet_module,
                           dif_module=dif_module,
                           data_loader=data_loader)

    trainer.train()
