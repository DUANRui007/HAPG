#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import time

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from models import GCN

import dgl
import torch
import pickle
import os
from adj_adjn_matrix import multi_hop

device = None
in_feats, n_classes = None, None
epsilon = 1 - math.log(2)  # ln2


def inverse_softmax(p, k):
    denominator = 0
    numerator = math.e ** -p
    for i in range(1, k+1):
        denominator += math.e ** -i
    order_attn = numerator / denominator
    return order_attn


def order_attn_(order_nums):
    order_attn = torch.zeros(order_nums)
    for i in range(order_nums):
        cur_order_attn = inverse_softmax(i + 1, order_nums)
        order_attn[i] = cur_order_attn
    return order_attn


def gen_graphs(args, g, order_nums, adjn_select, adj, dataset_name, path_2, path_3):
    g_list = []
    if args.loop:
       g = g.remove_self_loop().add_self_loop()
    g_list.append(g)

    if order_nums == 2:
        if os.path.isfile("./adj/%s_adj2__.pkl" % dataset_name):
            adj2 = torch.load("./adj/%s_adj2_.pkl" % dataset_name)
        elif os.path.isfile("./adj/simple_%s_adj2.pkl" % dataset_name):
            adj2 = torch.load("./adj/simple_%s_adj2.pkl" % dataset_name)
        else:
            adj2 = multi_hop(adj, order_nums)
        g2 = dgl.graph((adj2._indices()[0], adj2._indices()[1]), num_nodes=len(g.nodes()))

        if args.multi_loop:
            g2 = g2.remove_self_loop().add_self_loop()  # self loop
        g_list.append(g2)
        print('\n', '----------------A-MNS:2-hop------------------', '\n')

    if order_nums == 3:
        if os.path.isfile(".s/adj/%s_adj2__.pkl" % args.dataset_name):
            with open("./adj/%s_adj2__.pkl" % args.dataset_name, 'rb') as f:
                adj2 = pickle.load(f)
            f.close()
        if os.path.isfile("./adj/%s_adj3__.pkl" % args.dataset_name):
            with open("./adj/%s_adj3__.pkl" % args.dataset_name, 'rb') as f:
                adj3 = pickle.load(f)
            f.close()
        elif os.path.isfile("./adj/simple_%s_adj2.pkl" % dataset_name):
            adj2 = torch.load("./adj/simple_%s_adj2.pkl" % dataset_name)
            os.path.isfile("./adj/simple_%s_adj3.pkl" % dataset_name)
            adj3 = torch.load("./adj/simple_%s_adj3.pkl" % dataset_name)
        else:
            adj2, adj3 = multi_hop(adj, order_nums)
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
        values3 = adj3._values()
        values3_index = torch.nonzero(values3 > path_3, as_tuple=False).T.squeeze(0)
        indices3 = adj3._indices()[:, values3_index]
        g3 = dgl.graph((indices3[0], indices3[1]), num_nodes=len(g.nodes())).to(device)
        g_list[2] = g3
        print(f'P-MNS samples 2-hop and 3-hop neighbors successful according to path>{path_2} and path>{path_3}!', '\n')

    return g_list


def gen_model(args):
    if args.use_labels:
        model = GCN(
            in_feats + n_classes, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.use_linear
        )
    else:
        model = GCN(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.use_linear)
    return model


def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]


def add_labels(feat, labels, idx):
    onehot = th.zeros([feat.shape[0], n_classes]).to(device)
    onehot[idx, labels[idx, 0]] = 1
    return th.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(model, graph, labels, train_idx, optimizer, use_labels, order_attn):
    model.train()

    feat = graph[0].ndata["feat"]

    if use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat, order_attn)
    loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
    loss.backward()
    optimizer.step()

    return loss, pred


@th.no_grad()
def evaluate(model, graph, labels, train_idx, val_idx, test_idx, use_labels, evaluator, order_attn):
    model.eval()

    feat = graph[0].ndata["feat"]

    if use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat, order_attn)
    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running, order_attn):
    # define model and optimizer
    model = gen_model(args)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=1e-3
    )

    # training loop
    total_time = 0
    bad_epoch = 0
    best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")
    final_pred = None

    me_test_acc = 0

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        adjust_learning_rate(optimizer, args.lr, epoch)

        loss, pred = train(model, graph, labels, train_idx, optimizer, args.use_labels, order_attn)
        acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss = evaluate(
            model, graph, labels, train_idx, val_idx, test_idx, args.use_labels, evaluator, order_attn
        )

        lr_scheduler.step(loss)

        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc

            bad_epoch = 0
            best_test_acc = test_acc
            final_pred = pred
            val_epoch = epoch

        if val_loss > best_val_loss:
            bad_epoch += 1

        if test_acc > me_test_acc:
            me_test_acc = test_acc
            test_epoch = epoch
            me_val_loss = val_loss
            me_val_acc = val_acc

        if bad_epoch > args.patience:
            break

        for param in optimizer.param_groups:
            lrr = param['lr']

        if epoch % args.log_every == 0:
            print('-' * 70, 'lr=', lrr)
            print(f"Run: {n_running+1}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, ----Time:{time.time()-tic}")
            print(
                f"Loss: {loss.item():.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Best test acc: {train_acc:.4f}/{val_acc:.4f}/--{test_acc:.4f}/{best_val_acc:.4f}/--{best_test_acc:.4f}"
            )

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
        ):
            l.append(e)

    # torch.save(final_pred, './outputs/arxiv/gcn_LL/best_pred_run%s.pkl' % (n_running+1))

    print("*" * 50)
    print(f"Average epoch time: {total_time / args.n_epochs}, Test acc: {best_test_acc}")

    if args.plot_curves:
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.set_yticks(np.linspace(0, 1.0, 101))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip([accs, train_accs, val_accs, test_accs], ["acc", "train acc", "val acc", "test acc"]):
            plt.plot(range(args.n_epochs), y, label=label)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gcn_acc_{n_running}.png")

        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [losses, train_losses, val_losses, test_losses], ["loss", "train loss", "val loss", "test loss"]
        ):
            plt.plot(range(args.n_epochs), y, label=label)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gcn_loss_{n_running}.png")

    return best_val_acc, best_test_acc, final_pred, me_test_acc, val_epoch, test_epoch, best_val_loss, me_val_loss, me_val_acc


def count_parameters(args):
    model = gen_model(args)
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():
    global device, in_feats, n_classes

    argparser = argparse.ArgumentParser("GCN on OGBN-Arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=1, help="GPU device ID.")
    argparser.add_argument("--n-runs", type=int, default=2)
    argparser.add_argument("--n-epochs", type=int, default=1000)
    argparser.add_argument("--use_labels", type=bool, default=True, help="Use labels in the training set as input features")
    argparser.add_argument("--use_linear", type=bool, default=True, help="Use linear layer")
    argparser.add_argument('--loop', type=bool, default=True, help='need self loop?')
    argparser.add_argument('--multi_loop', type=bool, default=True, help='multi-hop need self loop?')
    argparser.add_argument("--lr", type=float, default=0.005)
    argparser.add_argument("--n-layers", type=int, default=3)
    argparser.add_argument("--n-hidden", type=int, default=256)
    argparser.add_argument("--dropout", type=float, default=0.5)

    argparser.add_argument('--dataset_name', type=str, default='arxiv', help='corafull arxiv')
    argparser.add_argument('--order_attn', type=bool, default=True, help='how set order attention')  #
    argparser.add_argument('--order_nums', type=int, default=2, help='Order number of need neighbour.')
    argparser.add_argument('--adjn_select', type=bool, default=True, help='select high-order adj element')
    argparser.add_argument('--path_2', type=int, default=2, help='select high-order adj element')
    argparser.add_argument('--path_3', type=int, default=1, help='select high-order adj element')
    argparser.add_argument('--patience', type=int, default=1000, help='patience')
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument("--log-every", type=int, default=10)
    argparser.add_argument("--plot-curves", action="store_true")
    args = argparser.parse_args()

    if args.cpu:
        device = th.device("cpu")
    else:
        device = th.device("cuda:%d" % args.gpu)

    # load data
    data = DglNodePropPredDataset(root="./OGB/dataset/", name="ogbn-arxiv")
    evaluator = Evaluator(name="ogbn-arxiv")

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
    adj = graph.adj()

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    # graph.create_format_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)

    # HAPG
    if args.order_attn:
        order_attn = order_attn_(args.order_nums)

    else:
        order_attn = []

    g_list_ = gen_graphs(args, graph, args.order_nums, args.adjn_select, adj, args.dataset_name, args.path_2, args.path_3)
    graph = []
    for i in range(len(g_list_)):
        graph.append(g_list_[i].to(device))

    # run
    val_epochs, test_epochs, val_losss, best_acc_losss,  me_test_accs = [], [], [], [], []  # me

    val_accs = []
    me_val_accs = []
    test_accs = []

    for i in range(args.n_runs):
        val_acc, test_acc, final_pred, me_test_acc, val_epoch, test_epoch, best_val_loss, me_val_loss, me_val_acc = \
            run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i, order_attn)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        val_epochs.append(val_epoch)
        test_epochs.append(test_epoch)
        val_losss.append(best_val_loss.item())
        best_acc_losss.append(me_val_loss.item())
        me_val_accs.append(me_val_acc)
        me_test_accs.append(me_test_acc)

        # torch.save(final_pred, './output/gcn_arxiv_mns%i_seed1.pkl' % i)

    print(f"Runned {args.n_runs} times")
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    print(f"Average test accuracy: {np.mean(me_test_accs)} ± {np.std(me_test_accs)}")
    print(f"Number of params: {count_parameters(args)}")

    print('-' * 70)
    print('best val loss epochs:', val_epochs)
    print('best val loss:', val_losss)
    print("Val Accs:", val_accs)
    print("test accs:", test_accs)
    print('-' * 70)
    print('best test acc epochs:', test_epochs)
    print('best test acc val_loss:', best_acc_losss)
    print("Val Accs:", me_val_accs)
    print('test accs:', me_test_accs)


if __name__ == "__main__":
    main()



