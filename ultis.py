import torch
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from SmilesEnumerator import SmilesEnumerator



from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def data_preprecess_task_1(root_dir='data', dataset='bTox', split='train', tokenizer=None, batch_size=8, device='cuda'):
    file_path = root_dir + '/' + dataset + '_' + split + '.csv'
    df = pd.read_csv(file_path).to_numpy()
    canonicals = df[:, 0]
    labels = df[:, 1:].squeeze().astype('int')

    if split == 'train':
        canonicals, labels = data_augmentation_task_1(canonicals, labels)

    print(canonicals.shape)

    tokens= tokenizer.batch_encode_plus(
        canonicals.tolist(),
        max_length=80,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )
    # print(tokens)

    input_ids = torch.tensor(tokens['input_ids'])
    att_masks = torch.tensor(tokens['attention_mask'])
    labels = torch.FloatTensor(labels)

    dataset = TensorDataset(input_ids, att_masks, labels)
    data_loader = prepare_dataloader(dataset, batch_size)

    return data_loader

def data_augmentation_task_1(smiles, labels, k=20):
    sme = SmilesEnumerator()
    print(smiles.shape)
    total_smiles = []
    total_labels = []
    for i in range(smiles.shape[0]):
        smile = smiles[i]
        label = labels[i]
        aug_smiles = [smile]
        aug_labels = [label]
        for _ in range(k):
            new_smile = sme.randomize_smiles(smile)
            aug_smiles.append(new_smile)
            aug_labels.append(label)
        total_smiles += aug_smiles
        total_labels += aug_labels

    total_smiles = np.array(total_smiles)
    total_labels = np.array(total_labels)

    return total_smiles, total_labels

def data_preprecess_task_2(root_dir='data', dataset='rToxcast', split='train', tokenizer=None, batch_size=8, device='cuda'):
    file_path = root_dir + '/' + dataset + '_' + split + '.csv'
    df = pd.read_csv(file_path)
    df.fillna(0.5, inplace=True)  # fillna with -1
    df = df.to_numpy()

    canonicals = df[:, 0]
    labels = df[:, 1:].astype('float')
    weight = (labels != 0.5).astype('float')

    if split == 'train':
        canonicals, labels, weight = data_augmentation_task_2(canonicals, labels, weight)

    print(canonicals.shape)

    tokens= tokenizer.batch_encode_plus(
        canonicals.tolist(),
        max_length=80,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )
    # print(tokens)

    input_ids = torch.tensor(tokens['input_ids'])
    att_masks = torch.tensor(tokens['attention_mask'])
    labels = torch.FloatTensor(labels)
    weight = torch.FloatTensor(weight)

    dataset = TensorDataset(input_ids, att_masks, labels, weight)
    data_loader = prepare_dataloader(dataset, batch_size)

    return data_loader

def data_augmentation_task_2(smiles, labels, weight, k=20):
    sme = SmilesEnumerator()
    print(smiles.shape)
    total_smiles = []
    total_labels = []
    total_weight = []
    for i in range(smiles.shape[0]):
        smile_i = smiles[i]
        label_i = labels[i]
        weight_i = weight[i]
        aug_smiles = [smile_i]
        aug_labels = [label_i]
        aug_weights = [weight_i]
        for _ in range(k):
            new_smile = sme.randomize_smiles(smile_i)
            aug_smiles.append(new_smile)
            aug_labels.append(label_i)
            aug_weights.append(weight_i)
        total_smiles += aug_smiles
        total_labels += aug_labels
        total_weight += aug_weights

    total_smiles = np.array(total_smiles)
    total_labels = np.array(total_labels)
    total_weight = np.array(total_weight)

    return total_smiles, total_labels, total_weight


def prepare_dataloader(datasets, batch_size):
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
    return dataloader


def plot_molecule(nx_graph):
    layout = nx.spring_layout(nx_graph)
    elements = nx.get_node_attributes(nx_graph, 'element')
    nx.draw(nx_graph, with_labels=True, labels=elements, pos=layout)
    plt.show()


