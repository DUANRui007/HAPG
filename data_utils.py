import numpy as np
import scipy.sparse as sp
import torch
from math import e
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CiteseerGraphDataset, PubmedGraphDataset, CoraGraphDataset
import dgl


def inverse_softmax(p, k):
    denominator = 0
    numerator = e ** -p
    for i in range(1, k+1):
        denominator += e ** -i
    order_attn = numerator / denominator
    return order_attn


def accuracy(output, labels):
    preds = output.max(dim=1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return (correct / len(labels)) * 100


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora_/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_data_(dataset_name, dataset_type):
    if dataset_type == 'dgl' and dataset_name == 'citeseer':
        dataset = CiteseerGraphDataset('./data/citeseer/')
    if dataset_type == 'dgl' and dataset_name == 'pubmed':
        dataset = PubmedGraphDataset('./data/pubmed/')

    if dataset_type == 'ogb' and dataset_name == 'arxiv':
        dataset = DglNodePropPredDataset(root='./OGB/dataset', name='ogbn-arxiv')

    if dataset_type == 'dgl' and dataset_name != 'cora':
        g = dataset[0]
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']

    if dataset_type == 'ogb':
        g, labels = dataset[0]
        src, dst = g.all_edges()
        g.add_edges(dst, src)
        split_idx = dataset.get_idx_split()
        features = g.ndata['feat']
        labels = labels  # .T.squeeze(0)
        train_mask, val_mask, test_mask = split_idx["train"], split_idx["valid"], split_idx["test"]  # 90941 29799 48603

    return g, features, labels, train_mask, val_mask, test_mask


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 每行的和
    r_inv = np.power(rowsum, -1).flatten()  # 次方
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)  # 用每行和的倒数乘以原矩阵中的每个数
    return mx

