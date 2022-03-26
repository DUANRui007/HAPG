import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
# from dgl.nn.functional import edge_softmax
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.conv import SAGEConv  # , GATConv
import torch.nn.functional as F


class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class SAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, aggregator_type):
        super(SAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggregator_type=aggregator_type))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type=aggregator_type))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, output_dim, aggregator_type=aggregator_type))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, g_list, features, order_attn):
        if len(g_list) == 1:
            x = features
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(g_list[0], x)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](g_list[0], x)
            return x.log_softmax(dim=-1)

        if len(g_list) > 1:
            x = features

            x1 = self.convs[-3](g_list[0], x)
            x2 = self.convs[-3](g_list[1], x)
            if len(g_list) == 3:
                x3 = self.convs[-3](g_list[2], x)
                x = order_attn[0] * x1 + order_attn[1] * x2 + order_attn[2] * x3
            else:
                x = order_attn[0] * x1 + order_attn[1] * x2
            x = self.bns[-2](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.convs[-2](g_list[0], x)
            x = self.bns[-1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.convs[-1](g_list[0], x)

            return x.log_softmax(dim=-1)

        # if len(g_list) > 1:
        #     x = features
        #     for i, conv in enumerate(self.convs[:-1]):
        #         x1 = conv(g_list[0], x)
        #         # x1 = self.bns[i](x1)
        #         # x1 = F.relu(order_attn[0] * x1)
        #         x1 = order_attn[0] * x1
        #
        #         x2 = conv(g_list[1], x)
        #         # x2 = self.bns[i](x2)
        #         # x2 = F.relu(order_attn[1] * x2)
        #         x2 = order_attn[1] * x2
        #         xx = x1 + x2
        #
        #         if len(g_list) == 3:
        #             x3 = conv(g_list[2], x)
        #             # x3 = self.bns[i](x3)
        #             # x3 = F.relu(order_attn[2] * x3)
        #             x3 = order_attn[2] * x3
        #             xx = x1 + x2 + x3
        #
        #         x = self.bns[i](xx)
        #         x = F.relu(x)
        #         x = F.dropout(x, p=self.dropout, training=self.training)
        #
        #     x = self.convs[-1](g_list[0], x)
        #     return x.log_softmax(dim=-1)


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, use_linear):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(dglnn.GraphConv(in_hidden, out_hidden, 'both', bias=bias, allow_zero_in_degree=True))  # "both"
            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_hidden))

        self.dropout0 = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat, order_attn):
        if len(graph) == 1:
            h = feat
            h = self.dropout0(h)  # 特征DROPOUT

            for i in range(self.n_layers):
                conv = self.convs[i](graph[0], h)

                if self.use_linear:
                    linear = self.linear[i](h)
                    h = conv + linear
                else:
                    h = conv

                if i < self.n_layers - 1:
                    h = self.bns[i](h)
                    h = self.activation(h)
                    h = self.dropout(h)

            return h

        if len(graph) > 1:
            h = feat
            h = self.dropout0(h)

            for i, conv_layer in enumerate(self.convs[:-2]):
                conv1 = self.convs[i](graph[0], h)
                conv2 = self.convs[i](graph[1], h)

                conv = order_attn[0] * conv1 + conv2 * order_attn[1]
                if len(graph) == 3:
                    conv3 = self.convs[i](graph[2], h)
                    conv = order_attn[0] * conv1 + conv2 * order_attn[1] + order_attn[2] * conv3
                if self.use_linear:
                    linear = self.linear[i](h)
                    h = conv + linear
                else:
                    h = conv

                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)

            conv_ = self.convs[-2](graph[0], h)
            if self.use_linear:
                linear_ = self.linear[-2](h)
                h = conv_ + linear_
            else:
                h = conv_
            h = self.bns[-1](h)
            h = self.activation(h)
            h = self.dropout(h)

            conv_ = self.convs[-1](graph[0], h)
            if self.use_linear:
                linear_ = self.linear[-1](h)
                h = conv_ + linear_
            else:
                h = conv_
            return h

        # if len(graph) > 1:
        #     h = feat
        #     h = self.dropout0(h)
        #
        #     for i, conv_layer in enumerate(self.convs[:-1]):
        #         conv1 = self.convs[i](graph[0], h)
        #         conv2 = self.convs[i](graph[1], h)
        #
        #         conv = order_attn[0] * conv1 + conv2 * order_attn[1]
        #         if len(graph) == 3:
        #             conv3 = self.convs[i](graph[2], h)
        #             conv = order_attn[0] * conv1 + conv2 * order_attn[1] + order_attn[2] * conv3
        #         if self.use_linear:
        #             linear = self.linear[i](h)
        #             h = conv + linear
        #         else:
        #             h = conv
        #
        #         h = self.bns[i](h)
        #         h = self.activation(h)
        #         h = self.dropout(h)
        #
        #     conv_ = self.convs[-1](graph[0], h)
        #     if self.use_linear:
        #         linear_ = self.linear[-1](h)
        #         h = conv_ + linear_
        #     else:
        #         h = conv_
        #     return h

            # for i in range(self.n_layers):
            #     conv1 = self.convs[i](graph[0], h)
            #     conv2 = self.convs[i](graph[1], h)
            #
            #     conv = order_attn[0] * conv1 + conv2 * order_attn[1]
            #     if len(graph) == 3:
            #         conv3 = self.convs[i](graph[2], h)
            #         conv = order_attn[0] * conv1 + conv2 * order_attn[1] + order_attn[2] * conv3
            #     if self.use_linear:
            #         linear = self.linear[i](h)
            #         h = conv + linear
            #     else:
            #         h = conv
            #
            #     if i < self.n_layers - 1:  # 非最后一层执行：
            #         h = self.bns[i](h)
            #         h = self.activation(h)
            #         h = self.dropout(h)
            #
            # return h


class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        use_attn_dst=True,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if use_attn_dst:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:
            self.register_buffer("attn_r", None)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if isinstance(self.attn_r, nn.Parameter):
            nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            if self.attn_r is not None:
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.dstdata.update({"er": er})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
            else:
                graph.apply_edges(fn.copy_u("el", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)  #
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:].to(e.device)
                graph.edata["a"] = torch.zeros_like(e).to(e.device)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class GAT(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        n_heads,
        activation,
        dropout=0.0,
        input_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        use_attn_dst=True,
        use_symmetric_norm=False,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            num_heads = n_heads if i < n_layers - 1 else 1
            out_channels = n_heads

            self.convs.append(
                GATConv(
                    in_hidden,
                    out_hidden,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    use_symmetric_norm=use_symmetric_norm,
                    residual=True,
                    allow_zero_in_degree=True
                )
            )

            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_channels * out_hidden))

        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat, order_attn):
        if len(graph) == 1:
            h = feat
            h = self.input_drop(h)

            for i in range(self.n_layers):
                conv = self.convs[i](graph[0], h)

                h = conv

                if i < self.n_layers - 1:
                    h = h.flatten(1)
                    h = self.norms[i](h)
                    h = self.activation(h, inplace=True)
                    h = self.dropout(h)

            h = h.mean(1)
            h = self.bias_last(h)

            return h

        if len(graph) > 1:
            h = feat
            h = self.input_drop(h)

            for i, conv_layer in enumerate(self.convs[:-2]):
                conv1 = self.convs[i](graph[0], h)
                conv2 = self.convs[i](graph[1], h)
                conv = order_attn[0] * conv1 + conv2 * order_attn[1]

                if len(graph) == 3:
                    conv3 = self.convs[i](graph[2], h)
                    conv = order_attn[0] * conv1 + conv2 * order_attn[1] + order_attn[2] * conv3

                h = conv

                h = h.flatten(1)
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

            h = self.convs[-2](graph[0], h)
            h = h.flatten(1)
            h = self.norms[-1](h)
            h = self.activation(h)
            h = self.dropout(h)

            h = self.convs[-1](graph[0], h)
            h = h.mean(1)
            h = self.bias_last(h)
            return h

        # if len(graph) > 1:
        #     h = feat
        #     h = self.input_drop(h)
        #
        #     for i, conv_layer in enumerate(self.convs[:-1]):
        #         conv1 = self.convs[i](graph[0], h)
        #         conv2 = self.convs[i](graph[1], h)
        #         conv = order_attn[0] * conv1 + conv2 * order_attn[1]
        #
        #         if len(graph) == 3:
        #             conv3 = self.convs[i](graph[2], h)
        #             conv = order_attn[0] * conv1 + conv2 * order_attn[1] + order_attn[2] * conv3
        #
        #         h = conv
        #
        #         h = h.flatten(1)
        #         h = self.norms[i](h)
        #         h = self.activation(h)
        #         h = self.dropout(h)
        #
        #     h = self.convs[-1](graph[0], h)
        #     h = h.mean(1)
        #     h = self.bias_last(h)
        #     return h

        # if len(graph) > 1:  # add code
        #     h = feat
        #     h = self.input_drop(h)
        #
        #     for i in range(self.n_layers):
        #         conv1 = self.convs[i](graph[0], h)
        #         conv2 = self.convs[i](graph[1], h)
        #         conv = order_attn[0] * conv1 + order_attn[1] * conv2
        #
        #         if len(graph) == 3:
        #             conv3 = self.convs[i](graph[2], h)
        #             conv = order_attn[0] * conv1 + order_attn[1] * conv2 + order_attn[2] * conv3
        #
        #         h = conv
        #
        #         if i < self.n_layers - 1:
        #             h = h.flatten(1)
        #             h = self.norms[i](h)
        #             h = self.activation(h, inplace=True)
        #             h = self.dropout(h)
        #
        #     h = h.mean(1)
        #     h = self.bias_last(h)
        #
        #     return h