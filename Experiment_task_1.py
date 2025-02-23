import torch
from ultis import data_preprecess_task_1
from models import BERT_Arch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from time import time
from config import get_config_task1

args = get_config_task1()
device = args.device
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr

pre_trained_model_name = 'seyonec/ChemBERTa_zi' \
                         'nc250k_v2_40k'
bert_tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
bert_model = AutoModel.from_pretrained(pre_trained_model_name, return_dict=True).to(device)

if __name__ == '__main__':
    loader_train = data_preprecess_task_1('data', 'bTox', 'train', tokenizer=bert_tokenizer, batch_size=batch_size, device=device)
    loader_val = data_preprecess_task_1('data', 'bTox', 'valid', tokenizer=bert_tokenizer, batch_size=512, device=device)
    loader_test = data_preprecess_task_1('data', 'bTox', 'test', tokenizer=bert_tokenizer, batch_size=512, device=device)

    num_classes = 1

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
        model = BERT_Arch(bert_model).to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2,
                                                    num_training_steps=20)
        t_0_train = time()
        for epoch in range(num_epochs):
            scheduler.step()
            model.train()
            total_acc_train = 0
            total_loss_train = 0
            for batch_idx, batch in enumerate(loader_train):
                batch = [r.to(device) for r in batch]
                input_ids_train, att_masks_train, labels_train = batch
                pred_train = model(input_ids_train, att_masks_train).squeeze()
                loss_train = F.binary_cross_entropy(pred_train, labels_train)
                loss_train.backward()
                optimizer.step()
                optimizer.zero_grad()
                acc_train = roc_auc_score(labels_train.cpu().numpy(), pred_train.detach().cpu().numpy())
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
                    input_ids_val, att_masks_val, labels_val = batch

                    pred_val = model(input_ids_val, att_masks_val).squeeze()
                    loss_val = F.binary_cross_entropy(pred_val, labels_val)
                    acc_val = roc_auc_score(labels_val.cpu().numpy(), pred_val.detach().cpu().numpy())
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
                    input_ids_test, att_masks_test, labels_test = batch
                    pred_test = model(input_ids_test, att_masks_test).squeeze()
                    loss_test = F.binary_cross_entropy(pred_test, labels_test)
                    # pred_label_test = pred_test.argmax(1).detach().cpu().numpy()
                    acc_test = roc_auc_score(labels_test.cpu().numpy(), pred_test.detach().cpu().numpy())
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
        torch.save(best_model.state_dict(), 'res/best_fine_tune_ChemBERTa_task1_seed_{}.pt'.format(seed))
        total_res.append(res)

    df = pd.DataFrame(total_res, columns=['Loss/Train', 'ROC-AUC/Train', 'Loss/Val', 'ROC-AUC/Val', 'Loss/Test', 'ROC-AUC/Test', 'Train_time', 'Infer_time'])
    df.to_excel('res/fine_tune_ChemBERTa_task1.xlsx')



