import numpy as np
import torch
from torch_geometric.loader import DataLoader
from ultis_gnn import data_preprecess_task_1
from models import GAT, GCN
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from time import time
from config import get_config_baseline_gnn_task1

args = get_config_baseline_gnn_task1()
device = args.device
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr
model_name = args.model_name

if __name__ == '__main__':
    writer = SummaryWriter('logs')

    device = 'cuda'
    datasets_train = data_preprecess_task_1('data', 'bTox', 'train')
    datasets_val = data_preprecess_task_1('data', 'bTox', 'valid')
    datasets_test = data_preprecess_task_1('data', 'bTox', 'test')

    loader_train = DataLoader(datasets_train, shuffle=True, batch_size=batch_size)
    loader_val = DataLoader(datasets_val, shuffle=False, batch_size=batch_size)
    loader_test = DataLoader(datasets_test, shuffle=False, batch_size=batch_size)

    dim_feat = 11
    num_classes = 1

    seeds = [1, 2, 3, 4, 5]
    total_res = []
    for seed in seeds:
        torch.manual_seed(seed)
        best_model = None
        best_loss_train = 0
        best_auc_train = 0
        best_loss_val = 0
        best_auc_val = 0
        best_loss_test = 0
        best_auc_test = 0
        if model_name == 'gat':
            model = GAT(dim_feat, [32, 64, 128, 256, 256, 128, 64, 32], num_classes, {'heads': 2}).to(device)
        elif model_name == 'gcn':
            model = GCN(dim_feat, [32, 64, 128, 256, 256, 128, 64, 32], num_classes, {'heads': 1}).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
        t_0_train = time()
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_auc = 0
            for batch_idx, (data_pyg_train, labels_train) in enumerate(loader_train):
                data_pyg_train = data_pyg_train.to(device)
                labels_train = labels_train.to(device)
                pred_train = model.forward(data_pyg_train.x, data_pyg_train.edge_index, data_pyg_train.batch).squeeze()
                loss_train = F.binary_cross_entropy(pred_train, labels_train.float())
                loss_train.backward()
                optimizer.step()
                optimizer.zero_grad()
                acc_train = roc_auc_score(labels_train.cpu().numpy(), pred_train.detach().cpu().numpy())
                total_loss += loss_train.item()
                total_auc += acc_train
                print('Train: epoch: {}, batch: {}, loss: {}, auc: {}'.format(epoch, batch_idx, loss_train, acc_train))
            total_loss = total_loss / len(loader_train)
            total_auc = total_auc / len(loader_train)
            writer.add_scalar('Loss/Train', total_loss, epoch)
            writer.add_scalar('ROC-AUC/Train', total_auc, epoch)

            is_best_model = False
            model.eval()
            for batch_idx, (data_pyg_val, labels_val) in enumerate(loader_val):
                data_pyg_val = data_pyg_val.to(device)
                labels_val = labels_val.to(device)
                pred_val = model.forward(data_pyg_val.x, data_pyg_val.edge_index, data_pyg_val.batch).squeeze()
                loss_val = F.binary_cross_entropy(pred_val, labels_val.float())
                acc_val = roc_auc_score(labels_val.cpu().numpy(), pred_val.detach().cpu().numpy())
                print("Valid: epoch: {}, loss: {}, auc: {}".format(epoch, loss_val, acc_val))

            if acc_val > best_auc_val:  # best_model
                is_best_model = True
                best_model = model
                best_loss_train = total_loss
                best_auc_train = total_auc
                best_loss_val = loss_val.item()
                best_auc_val = acc_val


            writer.add_scalar('Loss/Val', loss_val, epoch)
            writer.add_scalar('ROC-AUC/Val', acc_val, epoch)

                # Valid: epoch: 399, loss: 0.6446322202682495, acc: 0.6744186046511628

            t_0 = time()
            model.eval()
            for batch_idx, (data_pyg_test, labels_test) in enumerate(loader_test):
                data_pyg_test = data_pyg_test.to(device)
                labels_test = labels_test.to(device)
                pred_test = model.forward(data_pyg_test.x, data_pyg_test.edge_index, data_pyg_test.batch).squeeze()
                loss_test = F.binary_cross_entropy(pred_test, labels_test.float())
                acc_test = roc_auc_score(labels_test.cpu().numpy(), pred_test.detach().cpu().numpy())
            inference_time = time() - t_0
            print("Test: loss: {}, auc: {}, inference_time: {} s".format(loss_test, acc_test, inference_time))

            if is_best_model:
                best_loss_test = loss_test.item()
                best_auc_test = acc_test
            writer.add_scalar('Loss/Test', loss_test, epoch)
            writer.add_scalar('ROC-AUC/Test', acc_test, epoch)

        train_time = time() - t_0_train
        res = [best_loss_train, best_auc_train, best_loss_val, best_auc_val, best_loss_test, best_auc_test, train_time, inference_time]
        print(res)
        total_res.append(res)

        torch.save(best_model.state_dict(), 'res/best_{}_task1_seed_{}.pt'.format(model_name, seed))

    df = pd.DataFrame(total_res, columns=['Loss/Train', 'ROC-AUC/Train', 'Loss/Val', 'ROC-AUC/Val', 'Loss/Test', 'ROC-AUC/Test', 'Train_time', 'Infer_time'])
    df.to_excel('res/{}_task1.xlsx'.format(model_name))

    writer.close()


