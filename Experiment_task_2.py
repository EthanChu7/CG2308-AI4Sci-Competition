import torch
from ultis import data_preprecess_task_2
from models import BERT_Arch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from time import time
from config import get_config_task2


args = get_config_task2()
device = args.device
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr
pre_trained_model_name = 'seyonec/ChemBERTa_zinc250k_v2_40k'
bert_tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
bert_model = AutoModel.from_pretrained(pre_trained_model_name, return_dict=True).to(device)


batch_size = 256


def compute_roc_auc(labels, pred, weight):
    macro_auc = 0
    count = 617
    for i in range(617):
        weight_i = weight[:, i].bool()
        labels_i = labels[:, i][weight_i]
        pred_i = pred[:, i][weight_i]
        if labels_i.sum().item() == labels_i.shape[0] or labels_i.sum().item() == 0: # not valid for compute auc
            count -= 1
            continue
        auc = roc_auc_score(labels_i.detach().cpu().numpy(), pred_i.detach().cpu().numpy())
        macro_auc += auc
    return macro_auc / count

# def compute_loss(labels, pred, weight):
#     total_loss = 0
#     for i in range(617):
#         weight_i = weight[:, i].bool()
#         labels_i = labels[:, i][weight_i]
#         pred_i = pred[:, i][weight_i]
#         loss = F.binary_cross_entropy(pred_i, labels_i)
#         total_loss += loss
#     return total_loss / 617

def compute_loss(labels, pred, weight):
    return F.binary_cross_entropy(pred, labels, weight=weight)


if __name__ == '__main__':
    device = 'cuda'
    loader_train = data_preprecess_task_2('data', 'rToxcast', 'train', tokenizer=bert_tokenizer, batch_size=batch_size, device=device)
    loader_val = data_preprecess_task_2('data', 'rToxcast', 'valid', tokenizer=bert_tokenizer, batch_size=512, device=device)
    loader_test = data_preprecess_task_2('data', 'rToxcast', 'test', tokenizer=bert_tokenizer, batch_size=512, device=device)

    num_classes = 617
    model = BERT_Arch(bert_model, num_classes, dim_hidden=1024).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2,
                                                num_training_steps=20)


    seeds = [1, 2, 3, 4, 5]
    total_res = []
    for seed in seeds:
        print('seed', seed)
        torch.manual_seed(seed)
        best_model = None
        best_loss_train = 0
        best_auc_train = 0
        best_loss_val = 0
        best_auc_val = 0
        best_loss_test = 0
        best_auc_test = 0
        t_0_train = time()
        model = BERT_Arch(bert_model, num_classes, dim_hidden=1024).to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2,
                                                    num_training_steps=20)
        for epoch in range(num_epochs):
            scheduler.step()
            model.train()
            total_acc_train = 0
            total_loss_train = 0
            for batch_idx, batch in enumerate(loader_train):
                batch = [r.to(device) for r in batch]
                input_ids_train, att_masks_train, labels_train, weight_train = batch
                pred_train = model(input_ids_train, att_masks_train).squeeze()
                loss_train = compute_loss(labels_train, pred_train, weight_train)
                loss_train.backward()
                optimizer.step()
                optimizer.zero_grad()
                acc_train = compute_roc_auc(labels_train, pred_train, weight_train)
                total_loss_train += loss_train.item()
                total_acc_train += acc_train
                if batch_idx % 100 == 0:
                    print('Train: epoch: {}, batch: {}, loss: {}, auc: {}'.format(epoch, batch_idx, loss_train, acc_train))
            total_loss_train /= len(loader_train)
            total_acc_train /= len(loader_train)

            is_best_model = False
            with torch.inference_mode():
                model.eval()
                total_acc_val = 0
                total_loss_val = 0
                for batch_idx, batch in enumerate(loader_val):
                    batch = [r.to(device) for r in batch]
                    input_ids_val, att_masks_val, labels_val, weight_val = batch

                    pred_val = model(input_ids_val, att_masks_val).squeeze()
                    loss_val = compute_loss(labels_val, pred_val, weight_val)
                    acc_val = compute_roc_auc(labels_val, pred_val, weight_val)
                    total_acc_val += acc_val
                    total_loss_val += loss_val.item()
                total_acc_val /= len(loader_val)
                total_loss_val /= len(loader_val)
                print("Valid: epoch: {}, loss: {}, auc: {}".format(epoch, total_loss_val, total_acc_val))
                if total_acc_val > best_auc_val:
                    is_best_model = True
                    best_model = model
                    best_loss_train = total_loss_train
                    best_auc_train = total_acc_train
                    best_loss_val = total_loss_val
                    best_auc_val = total_acc_val

            t_0 = time()
            with torch.inference_mode():
                model.eval()
                total_acc_test = 0
                total_loss_test = 0
                for batch_idx, batch in enumerate(loader_test):
                    batch = [r.to(device) for r in batch]
                    input_ids_test, att_masks_test, labels_test, weight_test = batch
                    pred_test = model(input_ids_test, att_masks_test).squeeze()
                    loss_test = compute_loss(labels_test, pred_test, weight_test)
                    # pred_label_test = pred_test.argmax(1).detach().cpu().numpy()
                    acc_test = compute_roc_auc(labels_test, pred_test, weight_test)
                    total_acc_test += acc_test
                    total_loss_test += loss_test.item()
                inference_time = time() - t_0
                total_acc_test /= len(loader_test)
                total_loss_test /= len(loader_test)
                print("Test: loss: {}, auc: {}, inference_time: {} s".format(total_loss_test, total_acc_test, inference_time))
                if is_best_model:
                    best_loss_test = total_loss_test
                    best_auc_test = total_acc_test

        train_time = time() - t_0_train
        res = [best_loss_train, best_auc_train, best_loss_val, best_auc_val, best_loss_test, best_auc_test, train_time, inference_time]
        print(res)
        torch.save(best_model.state_dict(), 'res/best_fine_tune_ChemBERTa_task2_seed_{}.pt'.format(seed))
        total_res.append(res)

    df = pd.DataFrame(total_res, columns=['Loss/Train', 'ROC-AUC/Train', 'Loss/Val', 'ROC-AUC/Val', 'Loss/Test', 'ROC-AUC/Test', 'Train_time', 'Infer_time'])
    df.to_excel('res/fine_tune_ChemBERTa_task2.xlsx')



