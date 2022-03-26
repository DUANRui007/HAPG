import torch
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv, GraphConv, GINConv, APPNPConv, SAGEConv, AGNNConv, SGConv


class dgl_gat(nn.Module):
    def __init__(self, input_dim, out_dim, num_heads, num_classes, dropout, device):
        super(dgl_gat, self).__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device
        self.layer1 = GATConv(input_dim, out_dim, num_heads[0], feat_drop=dropout, attn_drop=dropout,
                              activation=None, allow_zero_in_degree=True, negative_slope=1.)
        self.layer2 = GATConv(num_heads[0] * out_dim, num_classes, num_heads[1], feat_drop=dropout, attn_drop=0.6,
                              activation=None, allow_zero_in_degree=True, negative_slope=1.)

    def forward(self, graph_list, input_features, order_attn):
        if len(graph_list) == 1:
            g1 = graph_list[0].to(self.device)
            x1 = self.layer1(g1, input_features)  # input_dim * num_heads[0] * out_dim
            x1 = x1.flatten(1)
            x1 = F.elu(x1)

            x1 = self.layer2(g1, x1)
            x1 = x1.squeeze(1)
            if self.num_heads[1] > 1:
                x1 = torch.mean(x1, dim=1)
            x1 = F.elu(x1)

            return x1

        if len(graph_list) == 2:
            g1 = graph_list[0].to(self.device)
            g2 = graph_list[1].to(self.device)

            x11 = self.layer1(g1, input_features)
            x11 = x11.flatten(1)
            x11 = F.elu(order_attn[0] * x11)
            x12 = self.layer1(g2, input_features)
            x12 = x12.flatten(1)
            x12 = F.elu(order_attn[1] * x12)
            x1 = x11 + x12

            x21 = self.layer2(g1, x1)
            x21 = x21.squeeze(1)
            x21 = F.elu(order_attn[0] * x21)
            if self.num_heads[1] > 1:
                x21 = torch.mean(x21, dim=1)

            if order_attn[1] > 0:
                x22 = self.layer2(g2, x1)
                x22 = x22.squeeze(1)
                x22 = F.elu(order_attn[1] * x22)
                if self.num_heads[1] > 1:
                    x22 = torch.mean(x22, dim=1)
            else:
                x22 = torch.zeros(x21.shape).to(x21.device)

            x2 = x21 + x22

            return x2

        if len(graph_list) == 3:
            g1 = graph_list[0].to(self.device)
            g2 = graph_list[1].to(self.device)
            g3 = graph_list[2].to(self.device)

            x11 = self.layer1(g1, input_features)
            x11 = x11.flatten(1)
            x11 = F.elu(order_attn[0] * x11)
            x12 = self.layer1(g2, input_features)
            x12 = x12.flatten(1)
            x12 = F.elu(order_attn[1] * x12)
            x13 = self.layer1(g3, input_features)
            x13 = x13.flatten(1)
            x13 = F.elu(order_attn[2] * x13)

            x1 = x11 + x12 + x13

            x21 = self.layer2(g1, x1)
            x21 = x21.squeeze(1)
            x21 = F.elu(order_attn[0] * x21)
            if self.num_heads[1] > 1:
                x21 = torch.mean(x21, dim=1)

            x22 = self.layer2(g2, x1)
            x22 = x22.squeeze(1)
            x22 = F.elu(order_attn[1] * x22)
            if self.num_heads[1] > 1:
                x22 = torch.mean(x22, dim=1)

            x23 = self.layer2(g3, x1)
            x23 = x23.squeeze(1)
            x23 = F.elu(order_attn[2] * x23)
            if self.num_heads[1] > 1:
                x23 = torch.mean(x23, dim=1)

            x2 = x21 + x22 + x23
            return x2


class dgl_gcn(nn.Module):
    def __init__(self, input_dim, nhidden, nclasses):
        super(dgl_gcn, self).__init__()
        self.layer1 = GraphConv(in_feats=input_dim, out_feats=nhidden, allow_zero_in_degree=True)
        self.layer2 = GraphConv(in_feats=nhidden, out_feats=nclasses, allow_zero_in_degree=True)

    def forward(self, g_list, features, order_attn):
        if len(g_list) == 1:
            g = g_list[0]
            x = self.layer1(g, features)
            x = self.layer2(g, x)
            return x

        if len(g_list) == 2:
            g1 = g_list[0]
            g2 = g_list[1].to(g1.device)
            x11 = self.layer1(g1, features)
            x12 = self.layer1(g2, features)
            x1 = order_attn[0] * x11 + order_attn[1] * x12

            x21 = self.layer2(g1, x1)
            x22 = self.layer2(g2, x1)
            x2 = order_attn[0] * x21 + order_attn[1] * x22
            return x2

        if len(g_list) == 3:
            g1 = g_list[0]
            g2 = g_list[1].to(g1.device)
            g3 = g_list[2].to(g1.device)
            x11 = self.layer1(g1, features)
            x12 = self.layer1(g2, features)
            x13 = self.layer1(g3, features)
            x1 = order_attn[0] * x11 + order_attn[1] * x12 + order_attn[2] * x13

            x21 = self.layer2(g1, x1)
            x22 = self.layer2(g2, x1)
            x23 = self.layer2(g3, x1)
            x2 = order_attn[0] * x21 + order_attn[1] * x22 + order_attn[2] * x23
            return x2


class dgl_sage(nn.Module):
    def __init__(self, input_dim, nhidden, aggregator_type, nclasses):
        super(dgl_sage, self).__init__()
        self.layer1 = SAGEConv(in_feats=input_dim, out_feats=nhidden, aggregator_type=aggregator_type)
        self.layer2 = SAGEConv(in_feats=nhidden, out_feats=nclasses, aggregator_type=aggregator_type)

    def forward(self, g_list, features, order_attn):
        if len(g_list) == 1:
            g = g_list[0]
            x = self.layer1(g, features)
            x = self.layer2(g, x)
            return x

        if len(g_list) == 2:
            g1 = g_list[0]
            g2 = g_list[1].to(g1.device)
            x11 = self.layer1(g1, features)
            x11 = order_attn[0] * x11
            x12 = self.layer1(g2, features)
            x12 = order_attn[1] * x12
            x1 = x11 + x12

            x21 = self.layer2(g1, x1)
            x21 = order_attn[0] * x21
            x22 = self.layer2(g2, x1)
            x22 = order_attn[1] * x22
            x2 = x21 + x22
            return x2

        if len(g_list) == 3:
            g1 = g_list[0]
            g2 = g_list[1].to(g1.device)
            g3 = g_list[2].to(g1.device)
            x11 = self.layer1(g1, features)
            x12 = self.layer1(g2, features)
            x13 = self.layer1(g3, features)
            x1 = order_attn[0] * x11 + order_attn[1] * x12 + order_attn[2] * x13

            x21 = self.layer2(g1, x1)
            x22 = self.layer2(g2, x1)
            x23 = self.layer2(g3, x1)
            x2 = order_attn[0] * x21 + order_attn[1] * x22 + order_attn[2] * x23
            return x2


class dgl_appnp(nn.Module):
    def __init__(self, input_dim, hidden, classes, k, alpha):
        super(dgl_appnp, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, classes)
        self.layer1 = APPNPConv(k=k, alpha=alpha, edge_drop=0.5)
        self.layer2 = APPNPConv(k=k, alpha=alpha, edge_drop=0.5)
        self.set_parameters()

    def set_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight, gain=gain)

    def forward(self, g_list, features, order_attn):
        features = self.fc1(features)
        if len(g_list) == 1:
            g = g_list[0]
            x = self.layer1(g, features)
            x = F.elu(self.fc2(x))
            x = self.layer2(g, x)
            x = F.elu(x)
            return x

        if len(g_list) == 2:
            g1 = g_list[0]
            g2 = g_list[1].to(g1.device)
            x11 = self.layer1(g1, features)
            x11 = F.elu(order_attn[0] * x11)
            x12 = self.layer1(g2, features)
            x12 = F.elu(order_attn[1] * x12)
            x1 = x11 + x12

            x1 = self.fc2(x1)
            x21 = self.layer2(g1, x1)
            x21 = F.elu(order_attn[0] * x21)
            x22 = self.layer2(g2, x1)
            x22 = F.elu(order_attn[1] * x22)
            x2 = x21 + x22
            return x2

        if len(g_list) == 3:
            g1 = g_list[0]
            g2 = g_list[1].to(g1.device)
            g3 = g_list[2].to(g1.device)
            x11 = self.layer1(g1, features)
            x11 = F.elu(order_attn[0] * x11)
            x12 = self.layer1(g2, features)
            x12 = F.elu(order_attn[1] * x12)
            x13 = self.layer1(g3, features)
            x13 = F.elu(order_attn[2] * x13)
            x1 = x11 + x12 + x13

            # x1 = F.elu(self.fc2(x1))
            x1 = self.fc2(x1)
            x21 = self.layer2(g1, x1)
            x21 = F.elu(order_attn[0] * x21)
            x22 = self.layer2(g2, x1)
            x22 = F.elu(order_attn[1] * x22)
            x23 = self.layer2(g3, x1)
            x23 = F.elu(order_attn[2] * x23)
            x2 = x21 + x22 + x23
            # x2 = F.elu(x2)
            return x2


class dgl_gin(nn.Module):
    def __init__(self, input_dim, hidden, classes, aggregator_type):
        super(dgl_gin, self).__init__()
        self.apply_func1 = nn.Linear(input_dim, hidden)
        self.apply_func2 = nn.Linear(hidden, classes)
        self.layer1 = GINConv(apply_func=self.apply_func1, aggregator_type=aggregator_type)
        self.layer2 = GINConv(apply_func=self.apply_func2, aggregator_type=aggregator_type)
        self.set_parameters()

    def set_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.apply_func1.weight, gain=gain)
        nn.init.xavier_normal_(self.apply_func2.weight, gain=gain)

    def forward(self, g_list, features, order_attn):
        if len(g_list) == 1:
            g = g_list[0]
            x = self.layer1(g, features)
            x = F.elu(x)
            x = self.layer2(g, x)
            x = F.elu(x)
            return x

        if len(g_list) == 2:
            g1 = g_list[0]
            g2 = g_list[1].to(g1.device)
            x11 = self.layer1(g1, features)
            x11 = F.elu(order_attn[0] * x11)
            x12 = self.layer1(g2, features)
            x12 = F.elu(order_attn[1] * x12)
            x1 = x11 + x12

            x21 = self.layer2(g1, x1)
            x21 = F.elu(order_attn[0] * x21)
            x22 = self.layer2(g2, x1)
            x22 = F.elu(order_attn[1] * x22)
            x2 = x21 + x22

            return x2

        if len(g_list) == 3:
            g1 = g_list[0]
            g2 = g_list[1].to(g1.device)
            g3 = g_list[2].to(g1.device)
            x11 = self.layer1(g1, features)
            x11 = F.elu(order_attn[0] * x11)
            x12 = self.layer1(g2, features)
            x12 = F.elu(order_attn[1] * x12)
            x13 = self.layer1(g3, features)
            x13 = F.elu(order_attn[2] * x13)
            x1 = x11 + x12 + x13

            x21 = self.layer2(g1, x1)
            x21 = F.elu(order_attn[0] * x21)
            x22 = self.layer2(g2, x1)
            x22 = F.elu(order_attn[1] * x22)
            x23 = self.layer2(g3, x1)
            x23 = F.elu(order_attn[2] * x23)
            x2 = x21 + x22 + x23
            return x2


class dgl_agnn(nn.Module):
    def __init__(self, input_dim, out_dim, num_classes, device, num_layers):
        super(dgl_agnn, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, num_classes)
        if num_layers == 2:
            self.layer1 = AGNNConv(init_beta=1., learn_beta=True, allow_zero_in_degree=True)
            self.layer2 = AGNNConv(init_beta=1., learn_beta=True, allow_zero_in_degree=True)
        if num_layers == 4:
            self.layer1 = AGNNConv(init_beta=1., learn_beta=True, allow_zero_in_degree=True)
            self.layer2 = AGNNConv(init_beta=1., learn_beta=True, allow_zero_in_degree=True)
            self.layer3 = AGNNConv(init_beta=1., learn_beta=True, allow_zero_in_degree=True)
            self.layer4 = AGNNConv(init_beta=1., learn_beta=True, allow_zero_in_degree=True)
        self.set_parameters()

    def set_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight, gain=gain)

    def forward(self, graph_list, input_features, order_attn):
        if len(graph_list) == 1:
            g1 = graph_list[0].to(self.device)
            if self.num_layers == 2:
                x1 = self.fc1(input_features)
                x1 = F.relu(x1)
                x1 = self.layer1(g1, x1)
                x1 = self.layer2(g1, x1)
                x1 = self.fc2(x1)
                F.log_softmax(x1, dim=1)
                return x1
            if self.num_layers == 4:
                x1 = self.fc1(input_features)
                x1 = F.relu(x1)
                x1 = self.layer1(g1, x1)
                x1 = self.layer2(g1, x1)
                x1 = self.layer3(g1, x1)
                x1 = self.layer4(g1, x1)
                x1 = self.fc2(x1)
                F.log_softmax(x1, dim=1)
                return x1

        if len(graph_list) == 2:
            g1 = graph_list[0].to(self.device)
            g2 = graph_list[1].to(self.device)
            if self.num_layers == 2:
                x1 = self.fc1(input_features)
                x1 = F.relu(x1)
                x11 = self.layer1(g1, x1)
                x11 = order_attn[0] * x11
                x12 = self.layer1(g2, x1)
                x12 = order_attn[1] * x12
                x2 = x11 + x12

                x21 = self.layer2(g1, x2)
                x21 = order_attn[0] * x21
                x22 = self.layer2(g2, x2)
                x22 = order_attn[1] * x22
                x3 = x21 + x22
                x3 = self.fc2(x3)
                F.log_softmax(x3)
                return x3

            if self.num_layers == 4:
                x1 = self.fc1(input_features)
                x1 = F.relu(x1)
                x11 = self.layer1(g1, x1)
                x12 = self.layer1(g2, x1)
                x1 = order_attn[0] * x11 + order_attn[1] * x12

                x21 = self.layer2(g1, x1)
                x22 = self.layer2(g2, x1)
                x2 = order_attn[0] * x21 + order_attn[1] * x22

                x31 = self.layer3(g1, x2)
                x32 = self.layer3(g2, x2)
                x3 = order_attn[0] * x31 + order_attn[1] * x32

                x41 = self.layer4(g1, x3)
                x42 = self.layer4(g2, x3)
                x4 = order_attn[0] * x41 + order_attn[1] * x42
                x4 = self.fc2(x4)
                F.log_softmax(x4)
                return x4

        if len(graph_list) == 3:
            g1 = graph_list[0].to(self.device)
            g2 = graph_list[1].to(self.device)
            g3 = graph_list[2].to(self.device)
            if self.num_layers == 2:
                x1 = self.fc1(input_features)
                x1 = F.relu(x1)
                x11 = self.layer1(g1, x1)
                x11 = order_attn[0] * x11
                x12 = self.layer1(g2, x1)
                x12 = order_attn[1] * x12
                x13 = self.layer1(g3, x1)
                x13 = order_attn[2] * x13
                x2 = x11 + x12 + x13

                x21 = self.layer2(g1, x2)
                x21 = order_attn[0] * x21
                x22 = self.layer2(g2, x2)
                x22 = order_attn[1] * x22
                x23 = self.layer2(g3, x2)
                x23 = order_attn[2] * x23
                x3 = x21 + x22 + x23
                x3 = self.fc2(x3)
                F.log_softmax(x3)
                return x3

            if self.num_layers == 4:
                x1 = self.fc1(input_features)
                x1 = F.relu(x1)
                x11 = self.layer1(g1, x1)
                x12 = self.layer1(g2, x1)
                x13 = self.layer1(g3, x1)
                x1 = order_attn[0] * x11 + order_attn[1] * x12 + order_attn[2] * x13

                x21 = self.layer2(g1, x1)
                x22 = self.layer2(g2, x1)
                x23 = self.layer2(g3, x1)
                x2 = order_attn[0] * x21 + order_attn[1] * x22 + order_attn[2] * x23

                x31 = self.layer3(g1, x2)
                x32 = self.layer3(g2, x2)
                x33 = self.layer3(g3, x2)
                x3 = order_attn[0] * x31 + order_attn[1] * x32 + order_attn[2] * x33

                x41 = self.layer4(g1, x3)
                x42 = self.layer4(g2, x3)
                x43 = self.layer4(g3, x3)
                x4 = order_attn[0] * x41 + order_attn[1] * x42 + order_attn[2] * x43
                x4 = self.fc2(x4)
                F.log_softmax(x4)
                return x4


class dgl_sgc(nn.Module):
    def __init__(self, input_dim, hidden, classes):
        super(dgl_sgc, self).__init__()
        self.layer1 = SGConv(in_feats=input_dim, out_feats=hidden, cached=False, allow_zero_in_degree=True)  # k=1
        self.layer2 = SGConv(in_feats=hidden, out_feats=classes, cached=False, allow_zero_in_degree=True)

    def forward(self, g_list, features, order_attn):
        if len(g_list) == 1:
            g = g_list[0]
            x = self.layer1(g, features)
            x = F.elu(x)  # elu(arxiv)//elu(cora...)
            x = self.layer2(g, x)
            return x

        if len(g_list) == 2:
            g1 = g_list[0]
            g2 = g_list[1].to(g1.device)
            x11 = self.layer1(g1, features)
            x11 = F.elu(order_attn[0] * x11)
            x12 = self.layer1(g2, features)
            x12 = F.elu(order_attn[1] * x12)
            x1 = x11 + x12

            x21 = self.layer2(g1, x1)
            x22 = self.layer2(g2, x1)

            x21 = order_attn[0] * x21
            x22 = order_attn[1] * x22
            x2 = x21 + x22
            return x2

        if len(g_list) == 3:
            g1 = g_list[0]
            g2 = g_list[1].to(g1.device)
            g3 = g_list[2].to(g1.device)
            x11 = self.layer1(g1, features)
            x11 = F.elu(order_attn[0] * x11)
            x12 = self.layer1(g2, features)
            x12 = F.elu(order_attn[1] * x12)
            x13 = self.layer1(g3, features)
            x13 = F.elu(order_attn[2] * x13)
            x1 = x11 + x12 + x13

            x21 = self.layer2(g1, x1)
            x21 = order_attn[0] * x21
            x22 = self.layer2(g2, x1)
            x22 = order_attn[1] * x22
            x23 = self.layer2(g3, x1)
            x23 = order_attn[2] * x23
            x2 = x21 + x22 + x23
            return x2
