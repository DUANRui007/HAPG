import torch
import scipy.sparse as sp
import numpy as np
import dgl


def multi_hop(g, num_hop):
    g = dgl.to_homogeneous(g)
    g2 = dgl.khop_graph(g, 2)
    adj2 = g2.adj().coalesce()
    indices = adj2._indices()
    values = adj2._values()

    indices = np.array(indices)
    csr_adj2 = sp.csr_matrix((values, (indices[0, :], indices[1, :])), shape=(len(g.nodes()), len(g.nodes())))
    csr_adj = g.add_self_loop().adj(scipy_fmt='csr')
    adj2_ = csr_adj2 - csr_adj * len(g.nodes())
    adj2_ = sp.coo_matrix(adj2_)

    row = torch.tensor(adj2_.row).unsqueeze(0)
    col = torch.tensor(adj2_.col).unsqueeze(0)
    values2 = torch.tensor(adj2_.data).float()

    values2_index = torch.nonzero(values2 > 0, as_tuple=True)
    values2 = values2[values2_index]
    indices__ = torch.cat([row[:, values2_index[0]], col[:, values2_index[0]]], dim=0)
    adj__ = torch.sparse_coo_tensor(indices__, values2)

    if num_hop == 2:
        return adj__

    if num_hop == 3:
        g3 = dgl.khop_graph(g, 3)
        adj3 = g3.adj().coalesce()
        indices = adj3._indices()
        values = adj3._values()
        indices = np.array(indices)
        scr_adj3 = sp.csr_matrix((values, (indices[0, :], indices[1, :])), shape=(len(g.nodes()), len(g.nodes())))
        indices = np.array(adj__._indices())
        values = np.array(adj__._values())

        scr_adj2 = sp.csr_matrix((values, (indices[0, :], indices[1, :])), shape=(len(g.nodes()), len(g.nodes())))
        adj3_ = scr_adj3 - csr_adj * len(g.nodes()) - scr_adj2 * len(g.nodes())
        adj3_ = sp.coo_matrix(adj3_)

        row = torch.tensor(adj3_.row).unsqueeze(0)
        col = torch.tensor(adj3_.col).unsqueeze(0)
        values3 = torch.tensor(adj3_.data).float()

        values3_index = torch.nonzero(values3 > 0, as_tuple=True)
        values3 = values3[values3_index]
        indices___ = torch.cat([row[:, values3_index[0]], col[:, values3_index[0]]], dim=0)
        adj___ = torch.sparse_coo_tensor(indices___, values3)
        return adj__, adj___


def adj_adj2(adj, device):  # tensor
    adj = adj.to(device)
    n = adj.shape[0]
    adj2 = torch.zeros(n, n).to(device)
    for i in range(n):
        neigh1 = torch.nonzero(adj[i], as_tuple=False)
        cur_adj2 = torch.sum(adj[neigh1], dim=0)
        cur_adj2 = cur_adj2 - adj[i] * n
        adj2[i] = cur_adj2
    zeros = torch.zeros(n, n).to(device)
    adje = torch.tensor(np.identity(n)).to(device)
    adj2 = torch.where(adje < 1, adj2, zeros)
    adj2 = torch.where(adj2 > 0, adj2, zeros)
    return adj2


def adj_adj3(adj, device):  # tensor
    adj = adj.to(device)
    n = adj.shape[0]
    adj2 = adj_adj2(adj, device)
    adj3 = torch.zeros(n, n).to(device)
    for i in range(n):
        neigh1 = torch.nonzero(adj[i], as_tuple=False)
        cur_adj3 = torch.sum(adj2[neigh1], dim=0)
        cur_adj3 = cur_adj3 - adj[i] * n - adj2[i] * n
        adj3[i] = cur_adj3
    zeros = torch.zeros(n, n).to(device)
    adje = torch.tensor(np.identity(n)).to(device)
    adj3 = torch.where(adje < 1, adj3, zeros)
    adj3 = torch.where(adj3 > 0, adj3, zeros)
    return adj2, adj3


def sp_adj_process(sp_adj, device):  # sp > tensor
    row = sp_adj.row
    col = sp_adj.col
    row = torch.tensor(row).to(device)
    col = torch.tensor(col).to(device)
    adj = torch.zeros(row.max()+1, row.max()+1).to(device)
    for i in range(len(row)):
        adj[row[i]][col[i]] = 1
    return adj


def sp_adj_adj2(sp_adj, device):  #
    sp_adj = sp_adj.to(device)
    n = sp_adj.shape[0]
    row = sp_adj._indices()[0]
    col = sp_adj._indices()[1]
    data = sp_adj._values()
    indices1 = []
    indices2 = []
    values = []
    for i in range(n):
        index1 = torch.nonzero(row == i, as_tuple=True)
        neigh1 = col[index1]
        for neg1 in neigh1:
            index2 = torch.nonzero(row == neg1.item(), as_tuple=True)
            neigh2 = col[index2]
            neigh2 = neigh2[neigh2 != i]
            neigh2 = torch.LongTensor([x for x in neigh2 if x not in neigh1])
            i_num = torch.zeros(len(neigh2)) + i
            indices1.extend(i_num)
            indices2.extend(neigh2)
            values.extend(data[neigh2])
    indices = torch.cat([torch.tensor(indices1).unsqueeze(0), torch.tensor(indices2).unsqueeze(0)], dim=0)
    indices = indices.to(device)
    values = torch.tensor(values)  # .cuda(0)  # .to(device)
    sp_adj2 = torch.sparse_coo_tensor(indices, values).to(device)
    return sp_adj2.coalesce()


def sp_adj_adj3(sp_adj, sp_adj2, device):
    sp_adj = sp_adj.to(device)
    sp_adj2 = sp_adj2.to(device)
    n = sp_adj.shape[0]
    row = sp_adj._indices()[0]
    col = sp_adj._indices()[1]
    row2 = sp_adj2._indices()[0]
    col2 = sp_adj2._indices()[1]
    data2 = sp_adj2._values()
    indices1 = []
    indices2 = []
    values = []
    for i in range(n):  # 50000-10000-150000-1,166,243
        index1 = torch.nonzero(row == i, as_tuple=False)
        neigh1 = col[index1]
        index2 = torch.nonzero(row2 == i, as_tuple=False)
        neigh2 = col2[index2]
        for neg1 in neigh1:
            index3 = torch.nonzero(row2 == neg1.item(), as_tuple=False)
            # print('index3:', sys.getsizeof(index3))
            for ind3 in index3:
                if col2[ind3] not in neigh1 and col2[ind3] not in neigh2:
                    indices1.append(i)
                    indices2.append(col2[ind3])
                    values.append(data2[ind3])
            del index3
        del index1
        del neigh1
    indices = torch.cat([torch.tensor(indices1).unsqueeze(0), torch.tensor(indices2).unsqueeze(0)], dim=0)
    indices = indices.to(device)
    values = torch.tensor(values).cuda(0)  # .to(device)
    sp_adj3 = torch.sparse_coo_tensor(indices, values)
    return sp_adj3.coalesce()

