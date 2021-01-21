import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F

from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj 

class KGATConv(MessagePassing):
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

        super(KGATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_dropout = node_dropout
        self.aggr_type = aggr_type
        assert self.aggr_type in ['GCN', 'GraphSAGE', 'Bi']

        # Equation 6 parameters
        if self.aggr_type == 'GCN':
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
            self.register_parameter('weight1', None)
            self.register_parameter('weight2', None)
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)
        
        # Equation 7 parameters
        elif self.aggr_type == 'GraphSAGE':
            self.weight = Parameter(torch.Tensor(2*in_channels, out_channels))
            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
            self.register_parameter('weight1', None)
            self.register_parameter('weight2', None)
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)
        
        # Equation 8 parameters
        elif self.aggr_type == 'Bi':
            self.weight1 = Parameter(torch.Tensor(in_channels, out_channels))
            self.weight2 = Parameter(torch.Tensor(in_channels, out_channels))
            if bias:
                self.bias1 = Parameter(torch.Tensor(out_channels))
                self.bias2 = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias1', None)
                self.register_parameter('bias2', None)
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        glorot(self.weight)
        zeros(self.bias1)
        zeros(self.bias2)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: Tensor) -> Tensor:
        """"""

        ego_embed = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        # Equation 6 w/o activation function
        if self.aggr_type == 'GCN':
            out = torch.matmul(x + ego_embed, self.weight)

            if self.bias is not None:
                out += self.bias

            return out
        
        # Equation 7 w/o activation function
        elif self.aggr_type == 'GraphSAGE':
            out = torch.matmul(torch.cat([x, ego_embed], dim=1), self.weight)

            if self.bias is not None:
                out += self.bias

            return out

        # Equation 8 w/o activation function
        elif self.aggr_type == 'Bi':
          out1 = torch.matmul(x + ego_embed, self.weight1)
          out2 = torch.matmul(x * ego_embed, self.weight2)

          if self.bias1 is not None:
                out1 += self.bias1
                out2 += self.bias2

          return (out1, out2)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        edge_weight = F.dropout(edge_weight, p=self.node_dropout, training=self.training)
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)