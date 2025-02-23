import torch
import pandas as pd
import numpy as np
from pysmiles import read_smiles
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

atom_pool = {
                'H':  [ 1, 2,  1,  1,  0,  0,  0,  0,  0],
                'Pt': [78, 7,  1,  2,  8, 18, 32, 17,  1],
                'Hg': [80, 7,  2,  2,  8, 18, 32, 18,  2],
                'B':  [ 5, 3,  3,  2,  3,  0,  0,  0,  0],
                'C':  [ 6, 3,  4,  2,  4,  0,  0,  0,  0],
                'Ge': [32, 5,  4,  2,  8, 18,  4,  0,  0],
                'N':  [ 7, 3,  5,  2,  5,  0,  0,  0,  0],
                'P':  [15, 4,  5,  2,  8,  5,  0,  0,  0],
                'As': [33, 5,  5,  2,  8, 18,  5,  0,  0],
                'O':  [ 8, 3,  6,  2,  6,  0,  0,  0,  0],
                'S':  [16, 4,  6,  2,  8,  6,  0,  0,  0],
                'F':  [ 9, 3,  7,  2,  7,  0,  0,  0,  0],
                'Cl': [17, 4,  7,  2,  8,  7,  0,  0,  0],
                'Br': [35, 5,  7,  2,  8, 18,  7,  0,  0],
                'I':  [53, 6,  7,  2,  8, 18, 18,  7,  0],
                'Ca': [20, 5,  2,  2,  8,  8,  2,  0,  0],
                'Na': [11, 4,  1,  2,  8,  1,  0,  0,  0],
                'Nd': [60, 6,  2,  2,  8,  18, 22, 8,  2],
                'Dy': [66, 6,  2,  2,  8,  18, 28, 8,  2],
                'In': [49, 5,  3,  2,  8,  18, 18, 3,  0],
    'Sb': [51, 5, 5, 2, 8, 18, 18, 5, 0],
    'Yb': [70, 6, 2, 2, 8, 18, 32, 8, 2],
    'Si': [14, 3, 4, 2, 8, 4, 0, 0, 0],
    'Sn': [50, 5, 4, 2, 8, 18, 18, 4, 0],
    'Mn': [25, 4, 2, 2, 8, 13, 2, 0, 0],
    'K': [19, 4, 1, 2, 8, 8, 1, 0, 0],
    'Au': [79, 6, 1, 2, 8, 18, 32, 18, 1],
    'Zn': [30, 4, 2, 2, 8, 18, 2, 0, 0],
    'Ni': [28, 4, 2, 2, 8, 16, 2, 0, 0],
    'Al': [13, 3, 3, 2, 8, 3, 0, 0, 0],
    'Fe': [26, 4, 2, 2, 8, 14, 2, 0, 0],
    'Ba': [56, 6, 2, 2, 8, 18, 18, 8, 2],
    'Ti': [22, 4, 2, 2, 8, 10, 2, 0, 0],
    'Pd': [46, 4, 18, 2, 8, 18, 18, 0, 0],
    'Cu': [29, 4, 1, 2, 8, 1, 0, 0, 0],
'Bi': [83, 6,  5,  2,  8,  18, 32, 18,  5], # DONTKNOW
'Mg': [12, 3,  2,  2,  8,  2, 0, 0,  0], # DONTKNOW

             }



def data_preprecess_task_1(root_dir='data', dataset='bTox', split='train'):
    file_path = root_dir + '/' + dataset + '_' + split + '.csv'
    df = pd.read_csv(file_path).to_numpy()
    # print(df.shape)
    canonicals = df[:, 0]

    labels = df[:, 1:].squeeze()
    # print(labels)

    # convert the smiles to nx graph
    # convert nx graph to pyg graph
    datasets = []
    elements = set()
    for idx, can_smile in enumerate(canonicals):
        molecule_nx = read_smiles(can_smile, explicit_hydrogen=True)
        features_element = np.array([atom_pool[node[1]] for node in molecule_nx.nodes('element')])
        [elements.add(node[1]) for node in molecule_nx.nodes('element')]
        features_aromatic_charge = np.array([[node[1]['charge'], node[1]['aromatic']] for node in molecule_nx.nodes(data=True)]).astype('float')
        features_node = np.concatenate([features_element, features_aromatic_charge], axis=1)
        node_list = molecule_nx.nodes
        adj = nx.adjacency_matrix(molecule_nx)
        edge_index, edge_weight = pyg.utils.convert.from_scipy_sparse_matrix(adj)
        molecule_pyg = Data(x=torch.FloatTensor(features_node), edge_index=edge_index, edge_weight=edge_weight)
        datasets.append((molecule_pyg, labels[idx]))

    return datasets
def data_preprecess_task_2(root_dir='data', dataset='rToxcast', split='train'):
    file_path = root_dir + '/' + dataset + '_' + split + '.csv'
    df = pd.read_csv(file_path)
    df.fillna(0.5, inplace=True) # fillna with -1
    df = df.to_numpy()
    # print(df.shape)
    canonicals = df[:, 0]
    labels = df[:, 1:].astype('float')
    weight = (labels!=0.5).astype('float')
    labels = torch.FloatTensor(labels)
    weight = torch.FloatTensor(weight)

    # convert the smiles to nx graph
    # convert nx graph to pyg graph
    datasets = []
    elements = set()
    for idx, can_smile in enumerate(canonicals):
        molecule_nx = read_smiles(can_smile, explicit_hydrogen=True)
        features_element = np.array([atom_pool[node[1]] for node in molecule_nx.nodes('element')])
        [elements.add(node[1]) for node in molecule_nx.nodes('element')]
        features_aromatic_charge = np.array([[node[1]['charge'], node[1]['aromatic']] for node in molecule_nx.nodes(data=True)]).astype('float')
        print(features_aromatic_charge.shape, features_element.shape)
        features_node = np.concatenate([features_element, features_aromatic_charge], axis=1)
        node_list = molecule_nx.nodes
        adj = nx.adjacency_matrix(molecule_nx)
        edge_index, edge_weight = pyg.utils.convert.from_scipy_sparse_matrix(adj)
        molecule_pyg = Data(x=torch.FloatTensor(features_node), edge_index=edge_index, edge_weight=edge_weight)
        datasets.append((molecule_pyg, labels[idx], weight[idx]))



    print(elements)

    return datasets

def data_append_representation(model, root_dir='data', dataset='bTox', split='valid'):
    datasets = data_preprecess_task_2(root_dir, dataset, split)
    loader = DataLoader(datasets, shuffle=False, batch_size=len(datasets))
    _, (data_pyg, label) = next(enumerate(loader))
    rep1, rep2 = model(data_pyg.x, data_pyg.edge_index, data_pyg.batch)
    rep = torch.concat([rep1, rep2], dim=1).detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    new_df = pd.DataFrame(np.concatenate([rep, np.expand_dims(label, axis=1)], axis=1))
    new_df.to_csv(root_dir + '/' + dataset + '_' + split + '_' + 'rep' + '.csv', index=False)


def plot_molecule(nx_graph):
    layout = nx.spring_layout(nx_graph)
    elements = nx.get_node_attributes(nx_graph, 'element')
    nx.draw(nx_graph, with_labels=True, labels=elements, pos=layout)
    plt.show()


#
# if __name__ == '__main__':
#     datasets = data_preprecess('data', 'rToxcast', 'test')
#     loader_train = DataLoader(datasets, shuffle=True, batch_size=2)
#     graph, y, weight = next(iter(loader_train))
#     print(graph)
#     print(y)
#     print(weight)
