# use dgl
# ogb dataset train > node classify

import os
import time
import math
import random
import numpy as np
import pickle
from data_utils import inverse_softmax, normalize_features, load_data, load_data_
from adj_adjn_matrix import sp_adj_adj2, sp_adj_adj3
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CiteseerGraphDataset, PubmedGraphDataset, CoraGraphDataset
import torch
import dgl
import scipy.sparse as sp
from dgl_models import dgl_gcn, dgl_gin, dgl_sage, dgl_appnp, dgl_gat, dgl_sgc, dgl_agnn
from adj_adjn_matrix import multi_hop


def select_model(model_name):
    if model_name == 'GAT':
        model = dgl_gat(input_dim=features.shape[1], out_dim=args.hidden, num_heads=[args.num_heads1, args.num_heads2],
                         num_classes=(int(labels.max())+1), dropout=args.dropout, device=device)
    if model_name == 'GCN':
        model = dgl_gcn(input_dim=features.shape[1], nhidden=args.hidden, nclasses=(int(labels.max())+1))
    if model_name == 'SAGE':
        model = dgl_sage(input_dim=features.shape[1], nhidden=args.hidden, aggregator_type=args.sage_agg_type,
                         nclasses=(int(labels.max())+1))
    if model_name == 'APPNP':
        model = dgl_appnp(input_dim=features.shape[1], hidden=args.appnp_hidden, classes=(int(labels.max())+1),
                          k=args.K, alpha=args.alpha)
    if model_name == 'GIN':
        model = dgl_gin(input_dim=features.shape[1], hidden=args.gin_hidden, classes=(int(labels.max())+1),
                        aggregator_type=args.gin_agg_type)
    if model_name == 'AGNN':
        model = dgl_agnn(input_dim=features.shape[1], out_dim=args.agnn_hidden, num_classes=(int(labels.max())+1),
                         device=device, num_layers=args.num_layers)  # # # #
    if model_name == 'SGC':
        model = dgl_sgc(input_dim=features.shape[1], hidden=args.hidden, classes=(int(labels.max()) + 1))
    return model


def gen_graphs(args, g, order_nums, adjn_select, adj, dataset_name, path_2, path_3):
    g_list = []
    if args.loop:
       g = g.remove_self_loop().add_self_loop()
    g_list.append(g)

    if order_nums == 2:
        if os.path.isfile("./adj/%s_adj2.pkl" % dataset_name):
            adj2 = torch.load("./adj/%s_adj2.pkl" % dataset_name)
        else:
            adj2 = multi_hop(g, order_nums)
        g2 = dgl.graph((adj2._indices()[0], adj2._indices()[1]), num_nodes=len(g.nodes()))

        if args.multi_loop:
            g2 = g2.remove_self_loop().add_self_loop()  # self loop
        g_list.append(g2)
        print('\n', '----------------A-MNS:2-hop------------------', '\n')

    if order_nums == 3:
        if os.path.isfile("./adj/%s_adj2.pkl" % dataset_name):
            adj2 = torch.load("./adj/%s_adj2.pkl" % dataset_name)
            os.path.isfile("./adj/%s_adj3.pkl" % dataset_name)
            adj3 = torch.load("./adj/%s_adj3.pkl" % dataset_name)
        else:
            adj2, adj3 = multi_hop(g, order_nums)
        g2 = dgl.graph((adj2._indices()[0], adj2._indices()[1]), num_nodes=len(g.nodes()))
        g3 = dgl.graph((adj3._indices()[0], adj3._indices()[1]), num_nodes=len(g.nodes()))

        if args.multi_loop:
            g2 = g2.remove_self_loop().add_self_loop().to(device)
            g3 = g3.remove_self_loop().add_self_loop().to(device)
        g_list.append(g2)
        g_list.append(g3)
        print('---------------A-MNS:2-hop and 3-hop------------------', '\n')

    if adjn_select and order_nums == 2:
        values2 = adj2._values()
        values2_index = torch.nonzero(values2 > path_2, as_tuple=False).T.squeeze(0)
        indices2 = adj2._indices()[:, values2_index]
        g2 = dgl.graph((indices2[0], indices2[1]), num_nodes=len(g.nodes())).to(device)
        g_list[1] = g2
        print('P-MNS samples 2-hop neighbors successful according to path>%s!' % path_2, '\n')

    if adjn_select and order_nums == 3:
        values2 = adj2._values()
        values2_index = torch.nonzero(values2 > path_2, as_tuple=False).T.squeeze(0)
        indices2 = adj2._indices()[:, values2_index]
        g2 = dgl.graph((indices2[0], indices2[1]), num_nodes=len(g.nodes())).to(device)
        g_list[1] = g2
        values3 = adj3._values()
        values3_index = torch.nonzero(values3 > path_3, as_tuple=False).T.squeeze(0)
        indices3 = adj3._indices()[:, values3_index]
        g3 = dgl.graph((indices3[0], indices3[1]), num_nodes=len(g.nodes())).to(device)
        g_list[2] = g3
        print(f'P-MNS samples 2-hop and 3-hop neighbors successful according to path>{path_2:1d} and path>{path_3:1d}!', '\n')

    return g_list


def order_attn_(order_nums):
    order_attn = torch.zeros(order_nums)
    for i in range(order_nums):
        cur_order_attn = inverse_softmax(i + 1, order_nums)
        order_attn[i] = cur_order_attn
    return order_attn


parser = argparse.ArgumentParser()

parser.add_argument('--order_attn', type=bool, default=True, help='how set order attention')  #
parser.add_argument('--order_nums', type=int, default=2, help='Order number of need neighbour.')
parser.add_argument('--adjn_select', type=bool, default=True, help='select high-order adj element')
parser.add_argument('--path_2', type=int, default=1, help='select high-order adj element')
parser.add_argument('--path_3', type=int, default=1, help='select high-order adj element')
parser.add_argument('--loop', type=bool, default=True, help='need self loop?')
parser.add_argument('--multi_loop', type=bool, default=True, help='multi-hop need self loop')
parser.add_argument('--feat_norm', type=bool, default=True, help='FEATURES NORM')
parser.add_argument('--model', type=str, default='GAT', help='GAT, GCN, GIN, APPNP, SAGE, AGNN, SGC, deep_gcn')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')

parser.add_argument('--dataset_type', type=str, default='dgl', help='dgl ogb')
parser.add_argument('--dataset_name', type=str, default='cora', help='cora citeseer pubmed arxiv')
parser.add_argument('--num_layers', type=int, default=2, help='AGNN:Number of layer.')  # cora:2,cite:4
parser.add_argument('--gin_agg_type', type=str, default='mean', help='gin:sum mean max')
parser.add_argument('--sage_agg_type', type=str, default='gcn', help='sage:mean, gcn, pool, lstm')
parser.add_argument('--K', type=int, default=10, help='APPNP inter')  # paper:10
parser.add_argument('--alpha', type=float, default=0.1, help='APPNP alpha')  # paper:0.1

parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')  # epoch gcn:small
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--agnn_hidden', type=int, default=16, help='AGNN:Number of hidden units.')
parser.add_argument('--appnp_hidden', type=int, default=8, help='APPNP:Number of hidden units.')
parser.add_argument('--gin_hidden', type=int, default=8, help='MoNet:Number of hidden units.')
parser.add_argument('--num_heads1', type=int, default=8, help='GAT:number of head')  #
parser.add_argument('--num_heads2', type=int, default=1, help='GAT:number of head')  # pub:8

parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')  # 0.6
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--patience', type=int, default=50, help='Patience')

args = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

if args.dataset_name == 'cora':
    adj, features, labels, train_mask, val_mask, test_mask = load_data(path="./data/cora/", dataset="cora")
    g = dgl.from_scipy(adj).to(device)
    if args.feat_norm:
        features = normalize_features(features)
        features = torch.FloatTensor(np.array(features.todense()))
else:
    g, features, labels, train_mask, val_mask, test_mask = load_data_(args.dataset_name, args.dataset_type)
    g = g.to(device)
    if args.feat_norm and args.dataset_type == 'dgl':
        features = sp.csr_matrix(np.array(features))
        features = normalize_features(features)
        features = torch.FloatTensor(np.array(features.todense()))

# data to device
features = features.to(device)
labels = labels.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)
print('\n', 'dgl graph generate successful!', '\n')
features, labels = Variable(features), Variable(labels)
if args.loop:
    g = g.remove_self_loop().add_self_loop()
adj = g.adj().to(device)
indices_e = torch.range(0, len(g.nodes())-1)
indices_e = torch.cat([indices_e.unsqueeze(0), indices_e.unsqueeze(0)], dim=0)
values_e = torch.ones(len(g.nodes()))
adj_e = torch.sparse_coo_tensor(indices_e, values_e).to(device)

if args.order_attn:
    order_attn = order_attn_(args.order_nums)

    order_attn = torch.tensor([1., 0.2])

else:
    order_attn = []

g_list_ = gen_graphs(args, g, args.order_nums, args.adjn_select, adj, args.dataset_name, args.path_2, args.path_3)
g_list = []
for i in range(len(g_list_)):
    g_list.append(g_list_[i].to(device))

test_accs = []
for i in range(args.seed, args.seed+10):
    # seed
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)

    model = select_model(args.model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    print('\n', 'model:', args.model, 'Start training...', '\n')

    t0 = time.time()
    bad_epoch, best_val_loss = 0, float("inf")
    for epoch in range(args.epochs):
        model.train()
        t1 = time.time()

        outputs = model(g_list, features, order_attn)
        outputs = F.log_softmax(outputs, dim=1)
        train_loss = F.cross_entropy(outputs[train_mask], labels[train_mask])
        train_pred = outputs[train_mask].max(dim=1)[1].type_as(labels[train_mask])
        correct = train_pred.eq(labels[train_mask]).double()
        correct = correct.sum()
        train_acc = (correct / len(labels[train_mask])) * 100

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()  # val
        with torch.no_grad():
            outputs = model(g_list, features, order_attn)
            outputs = F.log_softmax(outputs, dim=1)
            val_loss = F.cross_entropy(outputs[val_mask], labels[val_mask]).item()
            val_pred = outputs[val_mask].max(dim=1)[1].type_as(labels[val_mask])
            correct = val_pred.eq(labels[val_mask]).double()
            correct = correct.sum()
            val_acc = (correct / len(labels[val_mask])) * 100

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epoch = 0

                torch.save(model.state_dict(), f'{args.dataset_name}_model.pkl')

            else:
                bad_epoch += 1

            t2 = time.time()

        epoch_time = time.time() - t1
        if (epoch+1) % 10 == 0:
            print('Epoch: {:3d}'.format(epoch+1), 'Train loss: {:.4f}'.format(train_loss.item()),
                  '|Train accuracy: {:.2f}%'.format(train_acc), '||Val loss: {:.4f}'.format(val_loss),
                  '||Val accuracy: {:.2f}%'.format(val_acc), '||Time: {:.2f}'.format(epoch_time))

        if bad_epoch == args.patience:
            break

    model.load_state_dict(torch.load(f'{args.dataset_name}_model.pkl'))
    model.eval()  # test
    with torch.no_grad():
        outputs = model(g_list, features, order_attn)
        outputs = F.log_softmax(outputs, dim=1)
        test_loss = F.cross_entropy(outputs[test_mask], labels[test_mask]).item()
        test_pred = outputs[test_mask].max(dim=1)[1].type_as(labels[test_mask])
        correct = test_pred.eq(labels[test_mask]).double()
        correct = correct.sum()
        test_acc = (correct / len(labels[test_mask])) * 100
        test_accs.append(test_acc)

    _time = time.time() - t0
    print('\n', 'Test accuracy:', test_acc, 'Test loss:', test_loss)
    print('Time of training model:', _time)
    print('End of the training !')

print(test_accs)
print(f'Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}')


