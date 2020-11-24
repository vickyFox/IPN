import random
from collections import OrderedDict

from torchtools import *
from train import ModelTrainer
from data import MiniImagenetLoader, TieredImagenetLoader
from backbone.WideResNet import wide_res
from model import CCMNet
from local import DifNet


def test():
    tt.arg.test_model = 'asset/checkpoints/WRN_mini_5_5' if tt.arg.test_model is None else tt.arg.test_model
    tt.arg.device = 'cuda:0' if tt.arg.device is None else tt.arg.device
    # replace dataset_root with your own
    tt.arg.dataset_root = '/root/IPN/' if tt.arg.dataset_root is None else tt.arg.dataset_root
    tt.arg.dataset = 'mini' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 5 if tt.arg.num_shots is None else tt.arg.num_shots
    tt.arg.num_unlabeled = 0 if tt.arg.num_unlabeled is None else tt.arg.num_unlabeled
    tt.arg.meta_batch_size = 20 if tt.arg.meta_batch_size is None else tt.arg.meta_batch_size
    tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_gpus = 1 if tt.arg.num_gpus is None else tt.arg.num_gpus
    tt.arg.features = False

    tt.arg.num_ways_train = tt.arg.num_ways
    tt.arg.num_ways_test = tt.arg.num_ways

    tt.arg.num_shots_train = tt.arg.num_shots
    tt.arg.num_shots_test = tt.arg.num_shots

    # model parameter related
    tt.arg.emb_size = 640

    # train, test parameters
    tt.arg.train_iteration = 100000 if tt.arg.dataset == 'mini' else 200000
    tt.arg.test_iteration = 10000
    tt.arg.test_interval = 5000
    tt.arg.test_batch_size = 10
    tt.arg.log_step = 100

    tt.arg.lr = 1e-3
    tt.arg.grad_clip = 5
    tt.arg.weight_decay = 1e-6
    tt.arg.dec_lr = 15000 if tt.arg.dataset == 'mini' else 30000
    tt.arg.dropout = 0.1 if tt.arg.dataset == 'mini' else 0.0

    # set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    enc_module = wide_res(num_classes=64, remove_linear=True)
    ccmnet_module = CCMNet(in_features=tt.arg.emb_size, hidden_features=tt.arg.emb_size)
    dif_module = DifNet()

    if tt.arg.dataset == 'mini':
        test_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='test')
    elif tt.arg.dataset == 'tiered':
        test_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='test')
    else:
        print('Unknown dataset!')

    data_loader = {'test': test_loader}

    # create trainer
    tester = ModelTrainer(
        enc_module=enc_module,
        ccmnet_module=ccmnet_module,
        dif_module=dif_module,
        data_loader=data_loader)

    wrn_checkpoint = torch.load(tt.arg.test_model + '/pretrained_wrn.pth.tar')
    checkpoint = torch.load(tt.arg.test_model + '/model_best.pth.tar')
    state_dict = OrderedDict()
    for k in wrn_checkpoint['enc_module_state_dict']:
        name = k
        # loaded model is single GPU but we will train it in multiple GPUS!
        if name[:7] != 'module.' and torch.cuda.device_count() > 1:
            name = 'module.' + name  # add 'module'
        # loaded model is multiple GPUs but we will train it in single GPU!
        elif name[:7] == 'module.' and torch.cuda.device_count() == 1:
            name = k[7:]  # remove `module.`
        state_dict[name] = wrn_checkpoint['enc_module_state_dict'][k]
    tester.enc_module.load_state_dict(state_dict)
    print("load pre-trained enc_module done!")

    state_dict = OrderedDict()
    for k in checkpoint['ccmnet_module_state_dict']:
        name = k
        # loaded model is single GPU but we will train it in multiple GPUS!
        if name[:7] != 'module.' and torch.cuda.device_count() > 1:
            name = 'module.' + name  # add 'module'
        # loaded model is multiple GPUs but we will train it in single GPU!
        elif name[:7] == 'module.' and torch.cuda.device_count() == 1:
            name = k[7:]  # remove `module.`
        state_dict[name] = checkpoint['ccmnet_module_state_dict'][k]

    tester.ccmnet_module.load_state_dict(state_dict)
    print("load pre-trained ccmnet_module done!")

    tester.val_acc = checkpoint['val_acc']
    tester.global_step = checkpoint['iteration']
    print(tester.val_acc)
    print(tester.global_step)
    with torch.no_grad():
        tester.eval(partition='test')


if __name__ == '__main__':
    test()
