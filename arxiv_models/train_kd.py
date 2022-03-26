#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import os
import random
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from loss import loss_kd_only
from adj_adjn_matrix import multi_hop
import pickle

from models import GAT

epsilon = 1 - math.log(2)

device = None

dataset = "ogbn-arxiv"
n_node_feats, n_classes = 0, 0


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
        if os.path.isfile("./adj/simple_%s_adj2.pkl" % dataset_name):
            adj2 = torch.load("./adj/simple_%s_adj2.pkl" % dataset_name)
        else:
            adj2 = multi_hop(adj, order_nums)
        g2 = dgl.graph((adj2._indices()[0], adj2._indices()[1]), num_nodes=len(g.nodes()))

        if args.multi_loop:
            g2 = g2.remove_self_loop().add_self_loop()  # self loop
        g_list.append(g2)
        print('\n', '----------------A-MNS:2-hop------------------', '\n')

    if order_nums == 3:
        if os.path.isfile("./adj/%s_adj2__.pkl" % args.dataset_name):
            with open("./adj/%s_adj2__.pkl" % args.dataset_name, 'rb') as f:
                adj2 = pickle.load(f)
            f.close()
        if os.path.isfile("./adj/%s_adj3__.pkl" % args.dataset_name):
            with open("./adj/%s_adj3__.pkl" % args.dataset_name, 'rb') as f:
                adj3 = pickle.load(f)
            f.close()
        elif os.path.isfile("./adj/simple_%s_adj2.pkl" % dataset_name):
            adj2 = torch.load("./adj/simple_%s_adj2.pkl" % dataset_name)
            # os.path.isfile("/home/duanrui/dr_formal/use_ogb_datasets/adj/simple_%s_adj3.pkl" % dataset_name)
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
        print('P-MNS samples 2-hop and 3-hop neighbors successful according to path>%s and path>%s!' % path_2 % path_3, '\n')

    return g_list


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def load_data(dataset):
    global n_node_feats, n_classes

    data = DglNodePropPredDataset(root="./OGB/dataset/", name="ogbn-arxiv")
    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph):
    global n_node_feats

    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
    adj = graph.adj()

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph, adj


def gen_model(args):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats

    model = GAT(
        n_node_feats_,
        n_classes,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        attn_drop=args.attn_drop,
        edge_drop=args.edge_drop,
        use_attn_dst=not args.no_attn_dst,
        use_symmetric_norm=args.use_norm,
    )

    return model


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def add_labels(feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer, evaluator, teacher_output, order_attn):
    model.train()

    # alpha = args.alpha
    # temp = args.temp

    alpha = args.alpha
    temp = args.temp

    feat = graph[0].ndata["feat"]

    if args.use_labels:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat, order_attn)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([train_pred_idx, val_idx, test_idx])
        for _ in range(args.n_label_iters):
            pred = pred.detach()
            torch.cuda.empty_cache()
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, feat, order_attn)

    loss_gt = custom_loss_function(pred[train_pred_idx], labels[train_pred_idx])
    loss_kd = loss_kd_only(pred, teacher_output, temp)  # 教师模型的损失
    loss = loss_gt*(1-alpha) + loss_kd*alpha
    loss.backward()
    optimizer.step()

    return evaluator(pred[train_idx], labels[train_idx]), loss.item(),loss_gt.item(),loss_kd.item()


@torch.no_grad()
def evaluate(args, model, graph, labels, train_idx, val_idx, test_idx, evaluator, order_attn):
    model.eval()

    feat = graph[0].ndata["feat"]

    if args.use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat, order_attn)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([val_idx, test_idx])
        for _ in range(args.n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, feat, order_attn)

    train_loss = custom_loss_function(pred[train_idx], labels[train_idx])
    val_loss = custom_loss_function(pred[val_idx], labels[val_idx])
    test_loss = custom_loss_function(pred[test_idx], labels[test_idx])

    return (
        evaluator(pred[train_idx], labels[train_idx]),
        evaluator(pred[val_idx], labels[val_idx]),
        evaluator(pred[test_idx], labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
        pred,
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running, order_attn):
    evaluator_wrapper = lambda pred, labels: evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]

    # define model and optimizer
    model = gen_model(args).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training loop
    total_time = 0
    best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")
    patience = 0
    final_pred = None

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()
        teacher_output = torch.load("./outputs/hla_gat_kd/best_pred_run%i.pkl" % n_running)
        adjust_learning_rate(optimizer, args.lr, epoch)

        acc, loss, loss_gt, loss_kd = train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer,
                                            evaluator_wrapper,teacher_output, order_attn)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = evaluate(
            args, model, graph, labels, train_idx, val_idx, test_idx, evaluator_wrapper, order_attn
        )

        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            final_test_acc = test_acc
            final_pred = pred
            patience = 0

        else:
            patience += 1

        if epoch == args.n_epochs or epoch % args.log_every == 0:
            print(
                f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                f"Loss: {loss:.4f}, Loss_gt: {loss_gt:.4f}, Loss_kd: {loss_kd:.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
            )

        if patience > args.patience:
            break

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
        ):
            l.append(e)

    # torch.save(final_pred, "./outputs/hla_gat_kd/101kd_best_pred_run%i.pkl" % n_running)

    print("*" * 50)
    print(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
    print("*" * 50)

    # plot learning curves
    if args.plot_curves:
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.set_yticks(np.linspace(0, 1.0, 101))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip([accs, train_accs, val_accs, test_accs], ["acc", "train acc", "val acc", "test acc"]):
            plt.plot(range(args.n_epochs), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_acc_{n_running}.png")

        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [losses, train_losses, val_losses, test_losses], ["loss", "train loss", "val loss", "test loss"]
        ):
            plt.plot(range(args.n_epochs), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_loss_{n_running}.png")

    if args.save_pred:
        os.makedirs("./output_74.16", exist_ok=True)
        torch.save(final_pred.cpu(), f"./output_74.16/{n_running}.pt")

    return best_val_acc, final_test_acc


def count_parameters(args):
    model = gen_model(args)
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def main():
    global device, n_node_feats, n_classes, epsilon

    argparser = argparse.ArgumentParser(
        "GAT implementation on ogbn-arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=1, help="GPU device ID.")
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument("--n-runs", type=int, default=10, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=2000, help="number of epochs")

    argparser.add_argument("--use-labels", type=bool, default=True, help="Use labels in the training set as input features.")
    argparser.add_argument("--alpha", type=float, default=0.95, help="ratio of kd loss")  # 0.5
    argparser.add_argument("--temp", type=float, default=0.7, help="temperature of kd")  # 1.0
    argparser.add_argument("--n-label-iters", type=int, default=1, help="number of label iterations")
    argparser.add_argument("--mask-rate", type=float, default=0.5, help="mask rate")
    argparser.add_argument("--no-attn-dst", type=bool, default=True, help="Don't use attn_dst.")
    argparser.add_argument("--use-norm", type=bool, default=True, help="Use symmetrically normalized adjacency matrix.")
    argparser.add_argument("--lr", type=float, default=0.002, help="learning rate")

    argparser.add_argument('--dataset_name', type=str, default='arxiv', help='corafull arxiv')
    argparser.add_argument('--order_attn', type=bool, default=True, help='how set order attention')  #
    argparser.add_argument('--order_nums', type=int, default=2, help='Order number of need neighbour.')
    argparser.add_argument('--adjn_select', type=bool, default=True, help='select high-order adj element')
    argparser.add_argument('--path_2', type=int, default=1, help='select high-order adj element')
    argparser.add_argument('--path_3', type=int, default=1, help='select high-order adj element')
    argparser.add_argument('--analysis', type=bool, default=True, help='analysis find best model')
    argparser.add_argument('--loop', type=bool, default=True, help='need self loop?')
    argparser.add_argument('--multi_loop', type=bool, default=True, help='multi-hop need self loop?')
    argparser.add_argument('--patience', type=int, default=1000, help='Patience')

    argparser.add_argument("--n-layers", type=int, default=3, help="number of layers")
    argparser.add_argument("--n-heads", type=int, default=3, help="number of heads")
    argparser.add_argument("--n-hidden", type=int, default=250, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.75, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.25, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention drop rate")
    argparser.add_argument("--edge-drop", type=float, default=0.3, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--log-every", type=int, default=10, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot-curves", type=bool, default=False, help="plot learning curves")
    argparser.add_argument("--save-pred", type=bool, default=False, help="save final predictions")
    args = argparser.parse_args()

    if not args.use_labels and args.n_label_iters > 0:
        raise ValueError("'--use-labels' must be enabled when n_label_iters > 0")

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    # graph = preprocess(graph)

    graph, adj = preprocess(graph)

    # MNS
    if args.order_attn:  # me
        order_attn = order_attn_(args.order_nums)

        order_attn = torch.tensor([0.65, 0.35])  #

        #

    else:
        order_attn = []

    g_list_ = gen_graphs(args, graph, args.order_nums, args.adjn_select, adj, args.dataset_name, args.path_2,
                         args.path_3)
    graph = []
    # graph.append(g_list_[1])
    for i in range(len(g_list_)):
        graph.append(g_list_[i].to(device))  # me

    labels, train_idx, val_idx, test_idx = map(
        lambda x: x.to(device), (labels, train_idx, val_idx, test_idx)
    )

    # run
    val_accs, test_accs = [], []

    for i in range(args.n_runs):
        seed(args.seed + i)
        val_acc, test_acc = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i + 1, order_attn)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    print(args)
    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    print(f"Number of params: {count_parameters(args)}")


if __name__ == "__main__":
    main()


# Namespace(adjn_select=True, alpha=0.95, analysis=True, attn_drop=0.0, cpu=False, dataset_name='arxiv', dropout=0.75, edge_drop=0.3, gpu=0, input_drop=0.25, log_every=10, loop=True, lr=0.002, mask_rate=0.5, multi_loop=True, n_epochs=2000, n_heads=3, n_hidden=250, n_label_iters=1, n_layers=3, n_runs=10, no_attn_dst=True, order_attn=True, order_nums=2, path_2=1, path_3=1, patience=500, plot_curves=False, save_pred=False, seed=0, temp=0.7, use_labels=True, use_norm=True, wd=0)
# Runned 10 times
# Val Accs: [0.7515017282459143, 0.7510654719957045, 0.7528776133427296, 0.7501929594952851, 0.7531460787274741, 0.7511997046880767, 0.7524413570925199, 0.7521057753615893, 0.7515352864190074, 0.7513003792073559]
# Test Accs: [0.7423410077567228, 0.7419706602473098, 0.7428553792975742, 0.7413945641215562, 0.7439252721025451, 0.7403863959014876, 0.7419912351089439, 0.7402012221467811, 0.740921342303973, 0.7425879060963315]
# Average val accuracy: 0.7517366354575656 ± 0.0008560952392452631
# Average test accuracy: 0.7418574985083225 ± 0.0010985222871192993
# Number of params: 1441580