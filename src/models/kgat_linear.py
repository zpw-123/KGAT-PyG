import random
import numpy as np 

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_sparse import SparseTensor, masked_select_nnz
from torch_geometric.utils import softmax
from torch_geometric.typing import Adj, OptTensor, PairTensor

from conv import KGATConvLinear

def L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1) / 2)

class KGATLinear(torch.nn.Module):

    def __init__(self, data, args):

        super(KGATLinear, self).__init__()

        self._parse_args(args)
        
        self._parse_data(data)

        #--------Initialize Embeddings---------
        # 1. entity_user_embed
        if self.pretrain==1:
            # Dimensions of Pretrained Embeddings not matching!
            assert self.entity_user_emb_dim == self.user_item_pretrained.size(1)
            ckg_cf_part_embed = nn.Embedding.from_pretrained(self.user_item_pretrained, freeze=False)
            other_entities_embed = nn.Embedding(self.n_nodes_ckg - self.user_item_pretrained.size(0), self.entity_user_emb_dim)
            nn.init.xavier_uniform_(other_entities_embed.weight)
            merged = torch.cat([ckg_cf_part_embed.weight.cpu(), other_entities_embed.weight], dim=0)
            self.entity_user_embed = nn.Embedding.from_pretrained(merged, freeze=False)
        else:
            self.entity_user_embed = nn.Embedding(self.n_nodes_ckg, self.entity_user_emb_dim)
            nn.init.xavier_uniform_(self.entity_user_embed.weight)
        # 2. relation_embed
        self.relation_embed = nn.Embedding(self.n_relations_ckg, self.relation_emb_dim)
        nn.init.xavier_uniform_(self.relation_embed.weight)

        #--------Initialize W_R---------
        self.W_R = nn.Parameter(torch.Tensor(self.n_relations_ckg, self.entity_user_emb_dim, self.relation_emb_dim))
        torch.nn.init.xavier_uniform_(self.W_R)

        #--------Model Architecture--------
        self.conv_layer_dim_list = [self.entity_user_emb_dim] + self.conv_layer_dims

        self.convs = torch.nn.ModuleList()
        for layer in range(len(self.conv_layer_dim_list)-1):
            self.convs.append(
                KGATConvLinear(self.conv_layer_dim_list[layer], self.conv_layer_dim_list[layer+1], aggr_type=self.aggr_type, node_dropout=self.node_dropout)
                )

    def forward(self, edge_index):
      
        edge_index = edge_index
        edge_weight = self.attention
        x = self.entity_user_embed(torch.arange(self.n_nodes_ckg, device=edge_index.device))

        embeds_concat = [x]

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            if self.aggr_type == "Bi":
                x = F.leaky_relu(x[0], self.negative_slope) + F.leaky_relu(x[1], self.negative_slope)
            else:
                x = F.leaky_relu(x, self.negative_slope)
            x = F.dropout(x, self.message_dropout, training=self.training)
            x = F.normalize(x, p=2, dim=1)
            embeds_concat.append(x)

        # Equation (11) – Concatenates Embeddings (cf_concat_dim:= entity_user_emb_dim + out_channel_dim of all layers)
        embeds_concat = torch.cat(embeds_concat, dim=1) # (n_nodes_ckg, cf_concat_dim)

        return embeds_concat

    #------------------------------Parser–-----------------------------

    def _parse_args(self, args):
        
        self.conv_layer_dims = eval(args.conv_layer_dims)
        self.entity_user_emb_dim = args.entity_user_emb_dim
        self.relation_emb_dim = args.relation_emb_dim
        self.cf_loss_lambda = args.cf_loss_lambda
        self.ckg_loss_lambda = args.ckg_loss_lambda
        self.negative_slope = args.negative_slope
        self.aggr_type = args.aggr_type
        self.message_dropout = args.message_dropout
        self.node_dropout = args.node_dropout
        self.pretrain = args.pretrain

    def _parse_data(self, data):

        # Adjacency list; used for generating samples
        self.ckg_head_dict = data.ckg_head_dict
        self.cf_train_user_dict = data.cf_train_user_dict

        # Used for tensor initialization, look-ups and for-loops
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.n_nodes_ckg = data.n_nodes_ckg
        self.n_relations_ckg = data.n_relations_ckg

        # Some Helpers for Positive and Negative Sampling during Training
        self.n_train_cf = data.n_train_cf
        self.n_triples_ckg = data.n_triples_ckg

        # Get Attention Prior
        self.attention = data.edge_weights

        # Pretrained Embeddings 
        self.user_item_pretrained = data.user_item_pretrained

    #------------------------------CF Loader------------------------------
    # Note, we only draw a dummy batch here. 
    # By this approach, we make sure that we draw n_train_interactions positive and negative samples.
    # The real batch is drawn in _cf_sample().
    def cf_loader(self, **kwargs):
        return DataLoader(range(self.n_train_cf), collate_fn=self._cf_sample, num_workers=4, **kwargs)

    def _cf_pos_sample(self, user, num_samples = 1):

        pos_items = self.cf_train_user_dict[user]
        pos_items_samples = []
        while True:
            if len(pos_items_samples) == num_samples: break
            index = np.random.randint(low=0, high=len(pos_items), size=1)[0]
            pos_item_id = pos_items[index]

            if pos_item_id not in pos_items_samples:
                pos_items_samples.append(pos_item_id)

        return pos_items_samples

    def _cf_neg_sample(self, user, num_samples = 1):

        neg_items_samples = []
        while True:
            if len(neg_items_samples) == num_samples: break
            neg_item_id = np.random.randint(low=self.n_users, high=self.n_users+self.n_items, size=1)[0]
                
            if neg_item_id not in self.cf_train_user_dict[user] and neg_item_id not in neg_items_samples:
                neg_items_samples.append(neg_item_id)

        return neg_items_samples

    def _cf_sample(self, dummy_batch):
        batch_size = len(dummy_batch)

        # Get real batch
        exist_users = self.cf_train_user_dict.keys()
        if batch_size <= len(exist_users):
            batch_users = random.sample(exist_users, batch_size)
        else:
            batch_users = [random.choice(exist_users) for _ in range(batch_size)]

        # Draw Positive and Negative Samples
        batch_pos_items, batch_neg_items = [], []
        for user in batch_users:
            batch_pos_items += self._cf_pos_sample(user)
            batch_neg_items += self._cf_neg_sample(user)

        batch_users = torch.LongTensor(batch_users)
        batch_pos_items = torch.LongTensor(batch_pos_items)
        batch_neg_items = torch.LongTensor(batch_neg_items)

        return batch_users, batch_pos_items, batch_neg_items

    #------------------------------CKG Loader------------------------------
    # Note, we only draw a dummy batch here. 
    # By this approach, we make sure that we draw n_triples_ckg positive and negative samples.
    # The real batch is drawn in _ckg_sample().
    def ckg_loader(self, **kwargs):
        return DataLoader(range(self.n_triples_ckg), collate_fn=self._ckg_sample, num_workers=4, **kwargs)

    def _ckg_pos_sample(self, head, num_samples = 1):
        
        pos_triples = self.ckg_head_dict[head]
        relation_samples, pos_tail_samples = [], []
        while True:
            if len(relation_samples) == num_samples: break
            index = np.random.randint(low=0, high=len(pos_triples), size=1)[0]

            pos_tail = pos_triples[index][0]
            relation = pos_triples[index][1]

            if relation not in relation_samples and pos_tail not in pos_tail_samples:
                relation_samples.append(relation)
                pos_tail_samples.append(pos_tail)

        return relation_samples, pos_tail_samples

    def _ckg_neg_sample(self, head, relation, num_samples = 1):

        neg_tail_samples = []
        while True:
            if len(neg_tail_samples) == num_samples: break
            neg_tail = np.random.randint(low=0, high=self.n_nodes_ckg, size=1)[0]
                
            if (neg_tail, relation) not in self.ckg_head_dict[head] and neg_tail not in neg_tail_samples:
                neg_tail_samples.append(neg_tail)

        return neg_tail_samples

    def _ckg_sample(self, dummy_batch):
        batch_size = len(dummy_batch)

        # Get real batch
        exist_heads = self.ckg_head_dict.keys()
        if batch_size <= len(exist_heads):
            batch_heads = random.sample(exist_heads, batch_size)
        else:
            batch_heads = [random.choice(exist_heads) for _ in range(batch_size)]

        # Draw Positive and Negative Samples
        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for head in batch_heads:
            relation_samples, pos_tail_samples = self._ckg_pos_sample(head)
            batch_relation += relation_samples
            batch_pos_tail += pos_tail_samples

            relation = relation_samples[0]
            batch_neg_tail += self._ckg_neg_sample(head, relation)

        batch_heads = torch.LongTensor(batch_heads)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)

        return batch_heads, batch_relation, batch_pos_tail, batch_neg_tail

    #--------------------Attention Module--------------------
    def _att_score(self, edge_index, relation):

        # Equation (4) – Relational Attention Mechanism
        r_mul_h = torch.matmul(self.entity_user_embed(edge_index[0]), self.W_R[relation])                               # (n_edge, relation_dim)
        r_mul_t = torch.matmul(self.entity_user_embed(edge_index[1]), self.W_R[relation])                               # (n_edge, relation_dim)
        r_embed = self.relation_embed(torch.tensor(relation, device=edge_index.device))                                 # (relation_dim)

        # Dimension Tracking during the following transformations
        # r_mul_h: (n_edge, relation_dim) -unsqueeze(1)-> (n_edge,1,relation_dim)
        # torch.tanh(r_mul_h + r_embed): (n_edge, relation_dim) -unsqueeze(2)-> (n_edge,relation_dim,1)
        # attention_scores (n_edge, 1, 1) -squeeze()-> (n_edge) 
        attention_scores = torch.bmm(r_mul_t.unsqueeze(1), torch.tanh(r_mul_h + r_embed).unsqueeze(2)).squeeze()        # (n_edge)

        return attention_scores

    def update_attention(self, edge_index: Adj, edge_type: Tensor):

        attention_scores = torch.empty(edge_type.size(), device=edge_index.device)
        
        head_nodes = edge_index[0]
        for relation in range(self.n_relations_ckg):
            tmp_edge_index = edge_index[:, edge_type == relation]
            att_score = self._att_score(tmp_edge_index, relation)
            attention_scores[edge_type == relation] = att_score

        # Equation 5
        self.attention = softmax(attention_scores, head_nodes)  # (nnz)

    #------------------------------CF Loss------------------------------
    # Original: _build_loss_phase_I
    def calc_cf_loss(self, embeds_concat, user, pos_items, neg_items):
        """ Calculate regularized CF Loss
        Args:
            embeds_concat:  concatenated embeddings from the layer (n_nodes_ckg, cf_concat_dim)
            user:           batch_users of current batch           (cf_batch_size)
            pos_items:      batch_pos_items of current batch       (cf_batch_size)
            neg_items:      batch_neg_items of current batch       (cf_batch_size)
        Return:
            cf_loss
        """

        # Get embeddings of current batch
        user_embed = embeds_concat[user]                            # (cf_batch_size, cf_concat_dim)
        pos_item_embed = embeds_concat[pos_items]                   # (cf_batch_size, cf_concat_dim)
        neg_item_embed = embeds_concat[neg_items]                   # (cf_batch_size, cf_concat_dim)

        # Equation (12) – Predict Matching Score
        pos_score = torch.sum(user_embed * pos_item_embed, dim=1)   # (cf_batch_size)
        neg_score = torch.sum(user_embed * neg_item_embed, dim=1)   # (cf_batch_size)

        # Equation (13) – CF Loss
        cf_base_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_base_loss = torch.mean(cf_base_loss)
        cf_reg_loss = 0.5 * (L2_loss_mean(user_embed) + L2_loss_mean(pos_item_embed) + L2_loss_mean(neg_item_embed))
        cf_reg_loss = self.cf_loss_lambda * cf_reg_loss
        cf_loss = cf_base_loss + cf_reg_loss

        return cf_loss, cf_base_loss, cf_reg_loss

    #------------------------------CKG Loss------------------------------
    # Original: _build_loss_phase_II
    def ckg_loss(self, h, r, pos_t, neg_t):
        """ Calculate regularized CKG Loss
        Args:
            h (list):      batch_heads of current batch (ckg_batch_size)
            r (list):      batch_relation of current batch (ckg_batch_size)
            pos_t (list):  batch_pos_tail of current batch  (ckg_batch_size)
            neg_t (list):  batch_neg_tail of current batch (ckg_batch_size)
        Returns:
            ckg_loss
        """

        # Embedding Look-up
        r_embed = self.relation_embed(r)                 # (ckg_batch_size, relation_emb_dim)
        W_r = self.W_R[r]                                # (ckg_batch_size, entity_emb_dim, relation_emb_dim)

        # Perform some matrix multiplication operations
        h_embed = self.entity_user_embed(h)              # (ckg_batch_size, entity_emb_dim)
        pos_t_embed = self.entity_user_embed(pos_t)      # (ckg_batch_size, entity_emb_dim)
        neg_t_embed = self.entity_user_embed(neg_t)      # (ckg_batch_size, entity_emb_dim)

        # Dimension Tracking during the following transformations
        # h_embed: (ckg_batch_size,entity_emb_dim) -unsqueeze-> (ckg_batch_size,1,entity_emb_dim)
        # r_mul_h: (ckg_batch_size,1,entity_emb_dim) * (ckg_batch_size,entity_emb_dim,relation_emb_dim) -bmm-> (ckg_batch_size,1,relation_emb_dim)
        # r_mul_h: (ckg_batch_size,1,relation_emb_dim) -squeeze-> (ckg_batch_size,relation_emb_dim)
        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)             # (ckg_batch_size, relation_emb_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)     # (ckg_batch_size, relation_emb_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)     # (ckg_batch_size, relation_emb_dim)

        # Equation (1) – Energy Score
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (ckg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (ckg_batch_size)

        # Equation (2) – CKG Loss
        ckg_base_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        ckg_base_loss = torch.mean(ckg_base_loss)
        ckg_reg_loss = 0.5 * (L2_loss_mean(r_embed) + L2_loss_mean(r_mul_h) + L2_loss_mean(r_mul_pos_t) + L2_loss_mean(r_mul_neg_t))
        ckg_reg_loss = self.ckg_loss_lambda * ckg_reg_loss
        ckg_loss = ckg_base_loss + ckg_reg_loss

        return ckg_loss, ckg_base_loss, ckg_reg_loss