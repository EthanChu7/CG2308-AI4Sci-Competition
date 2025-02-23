import torch.nn.functional as F
import torch.nn as nn
import torch

from torch_geometric.nn import GCNConv, GATConv, BatchNorm
from torch_geometric.nn import global_mean_pool

class BERT_Arch(nn.Module):
    def __init__(self, bert, num_classes=1, dim_hidden=512):
        super(BERT_Arch, self).__init__()

        self.bert = bert
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768, dim_hidden)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(dim_hidden, num_classes)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        out= self.bert(sent_id, attention_mask=mask)
        pooled_out = out[1]

        x = self.fc1(pooled_out)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        x = F.sigmoid(x)

        return x


class GAT(torch.nn.Module):
    def __init__(self, input_size, hidden_channels, num_classes, conv_params={}):
        super(GAT, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = GATConv(
            input_size, hidden_channels[0], **conv_params)
        self.bn1 = BatchNorm(hidden_channels[0] * conv_params['heads'])

        self.conv2 = GATConv(
            hidden_channels[0] * conv_params['heads'], hidden_channels[1], **conv_params)
        self.bn2 = BatchNorm(hidden_channels[1] * conv_params['heads'])

        self.conv3 = GATConv(
            hidden_channels[1] * conv_params['heads'], hidden_channels[2], **conv_params)
        self.bn3 = BatchNorm(hidden_channels[2] * conv_params['heads'])

        self.conv4 = GATConv(
            hidden_channels[2] * conv_params['heads'], hidden_channels[3], **conv_params)
        self.bn4 = BatchNorm(hidden_channels[3] * conv_params['heads'])


        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels[3] * conv_params['heads'], 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes)
        )

    def forward(self, x, edge_index, batch=None, edge_col=None):
        # Node embedding
        h = self.bn1(self.conv1(x, edge_index, edge_col))
        h = h.relu()
        # h = F.dropout(h, p=0.5, training=self.training)

        h = self.bn2(self.conv2(h, edge_index, edge_col))
        h = h.relu()

        h = self.bn3(self.conv3(h, edge_index, edge_col))
        h = h.relu()
        # h = F.dropout(h, p=0.5, training=self.training)

        h = self.bn4(self.conv4(h, edge_index, edge_col))
        # h = h.relu()


        # Readout layer
        h = global_mean_pool(h, batch)
        h = F.dropout(h, p=0.5, training=self.training)

        # Final classifier
        # h = F.dropout(h, p=0.5, training=self.training)
        h = self.mlp(h)
        h = F.sigmoid(h)

        return h


class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_channels, num_classes, conv_params={}):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = GCNConv(
            input_size, hidden_channels[0], **conv_params)
        self.bn1 = BatchNorm(hidden_channels[0] * conv_params['heads'])

        self.conv2 = GCNConv(
            hidden_channels[0] * conv_params['heads'], hidden_channels[1], **conv_params)
        self.bn2 = BatchNorm(hidden_channels[1] * conv_params['heads'])

        self.conv3 = GCNConv(
            hidden_channels[1] * conv_params['heads'], hidden_channels[2], **conv_params)
        self.bn3 = BatchNorm(hidden_channels[2] * conv_params['heads'])

        self.conv4 = GCNConv(
            hidden_channels[2] * conv_params['heads'], hidden_channels[3], **conv_params)
        self.bn4 = BatchNorm(hidden_channels[3] * conv_params['heads'])


        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels[3] * conv_params['heads'], 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes)
        )

    def forward(self, x, edge_index, batch=None, edge_col=None):
        # Node embedding
        h = self.bn1(self.conv1(x, edge_index, edge_col))
        h = h.relu()
        # h = F.dropout(h, p=0.5, training=self.training)

        h = self.bn2(self.conv2(h, edge_index, edge_col))
        h = h.relu()

        h = self.bn3(self.conv3(h, edge_index, edge_col))
        h = h.relu()
        # h = F.dropout(h, p=0.5, training=self.training)

        h = self.bn4(self.conv4(h, edge_index, edge_col))
        # h = h.relu()


        # Readout layer
        h = global_mean_pool(h, batch)
        h = F.dropout(h, p=0.5, training=self.training)

        # Final classifier
        # h = F.dropout(h, p=0.5, training=self.training)
        h = self.mlp(h)
        h = F.sigmoid(h)

        return h