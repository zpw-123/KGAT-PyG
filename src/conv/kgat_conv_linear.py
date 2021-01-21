import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj 

class KGATConvLinear(MessagePassing):
    r"""The graph convolutional operator from the 
    `"KGAT: Knowledge Graph Attention Network for Recommendation"
    <https://arxiv.org/abs/1905.07854>` paper.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        node_dropout (float): Dropout Ratio.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        aggr_type (str): Choose one of the Aggregators 'GCN', 'GraphSAGE', 'Bi'
            (default 'Bi')
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels: int, out_channels: int, node_dropout: float = 0.,
                 bias: bool = True, aggr_type: str = 'Bi', **kwargs):

        super(KGATConvLinear, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_dropout = node_dropout
        self.aggr_type = aggr_type
        assert self.aggr_type in ['GCN', 'GraphSAGE', 'Bi']

        # Equation 6 parameters
        if self.aggr_type == 'GCN':
            self.W = nn.Linear(self.in_channels, self.out_channels)
            nn.init.xavier_uniform_(self.W.weight)
            self.W.bias.data.fill_(0)
        
        # Equation 7 parameters
        elif self.aggr_type == 'GraphSAGE':
            self.W = nn.Linear(2 * self.in_channels, self.out_channels)
            nn.init.xavier_uniform_(self.W.weight)
            self.W.bias.data.fill_(0)
        
        # Equation 8 parameters
        elif self.aggr_type == 'Bi':
            self.W1 = nn.Linear(self.in_channels, self.out_channels)      
            self.W2 = nn.Linear(self.in_channels, self.out_channels)
            nn.init.xavier_uniform_(self.W1.weight)
            self.W1.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.W2.weight)
            self.W2.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: Tensor) -> Tensor:
        """"""

        ego_embed = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        # Equation 6 w/o activation function
        if self.aggr_type == 'GCN':
            out = self.W(x + ego_embed)

            return out
        
        # Equation 7 w/o activation function
        elif self.aggr_type == 'GraphSAGE':
            out = self.W(torch.cat([x, ego_embed], dim=1))

            return out

        # Equation 8 w/o activation function
        elif self.aggr_type == 'Bi':
            out1 = self.W1(x + ego_embed)
            out2 = self.W2(x * ego_embed)

            return (out1, out2)
            

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        edge_weight = F.dropout(edge_weight, p=self.node_dropout, training=self.training)
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)