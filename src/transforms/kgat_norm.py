import torch
from torch_scatter import scatter_add

class KGATNorm(object):
    r"""Row-normalizes Relational Adjacency Matrix"""

    def __init__(self, adj_type='si'):
        self.adj_type = adj_type

    def __call__(self, data):
        
        m = data.n_nodes_ckg
        edge_weights = torch.empty(data.edge_type.size())

        if self.adj_type == 'si':
            for relation in range(data.n_relations_ckg):
                adj_t = data.edge_index[:, data.edge_type == relation]
                edge_weight = torch.ones(adj_t.size(1))

                deg = scatter_add(edge_weight, adj_t[0])
                deg_inv_sqrt = deg.pow_(-1)
                deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

                edge_weights[data.edge_type == relation] = deg_inv_sqrt[adj_t[0]] * edge_weight

        data.edge_weights = edge_weights

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)