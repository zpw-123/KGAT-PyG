import os
import os.path as osp
import shutil

import torch
import numpy as np
import collections

from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)

class KnowledgeAwareRecommendation(InMemoryDataset):
    r"""The Collaborative Knowledge Graphs "last-fm", "amazon-book" and "yelp2018"
    the `"KGAT: Knowledge Graph Attention Network for Recommendation"
    <https://arxiv.org/abs/1905.07854>` paper.

    Data Reqs:* All Indices are remapped [0, n_type]. 
              * No duplicates in kf_file and rating_file
              * Each user in test set is also in train set

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"last-fm"`, :obj:`"amazon-book"`, :obj:`"yelp2018"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, name, model, transform=None, pre_transform=None):
        assert name in ['last-fm', 'amazon-book', 'yelp2018']
        assert model in ['kgat']
        self.name = name
        self.model = model

        # Instructions:
        # Note, that even if the Dropbox link is showing dl=0, use dl=1
        # Also make sure to zip the files together and not the folder
        # Use 'zip -r data.zip . -x ".*" -x "__MACOSX"' in order to not create this __MACOSX file
        if self.name == 'last-fm':
            self.url = "https://www.dropbox.com/s/d1qm6zrmeggr570/last-fm.zip?dl=1"
        elif self.name == 'amazon-book':
            self.url = "https://www.dropbox.com/s/srrvivdzv0s9g2s/amazon-book.zip?dl=1"
        elif self.name == 'yelp2018':
            self.url = "https://www.dropbox.com/s/8pjlpr415fb8xqb/yelp2018.zip?dl=1"

        super(KnowledgeAwareRecommendation, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['train.txt', 'test.txt', 'item_list.txt', 'user_list.txt', 
                'kg_final.txt', 'entity_list.txt', 'relation_list.txt',
                'pretrain.npz'
               ]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):

        def _load_pretrain(pretrain_path):
            """ Load Collaborative Filtering dataset
            Args:
                pretrain_path (path): Path of pretrained embeddings
            Returns:
                pretrain_data (ndarray): Pretrained Embeddings (n_user + n_items, dim)
            """

            pretrain_data = np.load(pretrain_path)
            user_pre_embed = torch.tensor(pretrain_data['user_embed'], dtype=torch.float32)
            item_pre_embed = torch.tensor(pretrain_data['item_embed'], dtype=torch.float32)

            pretrain_data = torch.cat([user_pre_embed, item_pre_embed], dim=0)

            return pretrain_data

        def _load_cf_data(cf_file_path):
            """ Load Collaborative Filtering dataset
            Args:
                cf_file_path (path): Path of the dataset
            Returns:
                cf_data (ndarray): [User-Item]-Pairs
            """

            user_item_pairs = []
            lines = open(cf_file_path, 'r').readlines()
            for l in lines:
                tmp = l.strip()
                ids = [int(i) for i in tmp.split()]

                if len(ids) > 1:
                    user_id, pos_item_ids = ids[0], ids[1:]
                    pos_item_ids = list(set(pos_item_ids))

                    for item_id in pos_item_ids:
                        user_item_pairs.append([user_id, item_id])

            cf_data = np.array(user_item_pairs, dtype=np.int32) 

            return cf_data

        def _load_kg(kg_file_path):
            """ Load knowledge graph dataset and construct knowledge graph
            Args:
                kg_file_name (path): Path of the dataset
            Returns:
                kg_data (ndarray): Loaded knowledge graph with [head, relation, tail]-triples
            """

            kg_data = np.loadtxt(kg_file_path, dtype=np.int32)

            return kg_data

        def _create_ckg(cf_train_data, cf_test_data, kg_data, n_users, n_relations):
            """ Construct CKG by reversing relations and remapping items/entities
            Args:
                cf_train_data (ndarray): [User-Item]-Pairs of Train Set
                cf_test_data (ndarray): [User-Item]-Pairs of Test Set
                kg_data (ndarray): [Entity, Relation, Entity]-Triples
            Returns:
                ckg_data (ndarray): [User/Entity, Relation, Entity]-Triples
                ckg_head_dict (dict): Maps each head entity to a list of its corresponding (tail, relation) tuples (remapped)
                cf_train_user_dict (dict): Maps each train user to a list of its corresponding positive items (remapped)
                cf_test_user_dict (dict): Maps each test user to a list of its corresponding positive items (remapped)
            """

            # Create CF Triples of the CKG
            # Note: Only consists out of Train Users and Items
            ckg_cf_triples = []                           # (2 * n_train_cf, 3)
            for user, item in cf_train_data:
                item = item + n_users # Make sure to remap items!!
                ckg_cf_triples.append([user, item, 0])
                ckg_cf_triples.append([item, user, n_relations + 1])
        
            # Create CKG Triples of the CKG
            ckg_kg_triples = []                           # (2 * n_triples, 3)
            for head, relation, tail in kg_data:
                head = head + n_users # Make sure to remap entities!!
                tail = tail + n_users # Make sure to remap entities!!
                ckg_kg_triples.append([head, tail, relation + 1])
                ckg_kg_triples.append([tail, head, relation + 2 + n_relations])

            # Create full CKG 
            ckg_data = ckg_cf_triples + ckg_kg_triples    # (2 * n_train_cf + 2 * n_triples, 3)

            # Get Head-Lookup Dict for the CKG
            ckg_head_dict = collections.defaultdict(list)
            for head, tail, relation in ckg_data:
                ckg_head_dict[head].append((tail, relation))

            # Get User-Lookup Dict for the CF
            cf_train_user_dict = collections.defaultdict(list)
            for user, item in cf_train_data:
                item = item + n_users # Make sure to remap items!!
                cf_train_user_dict[user].append(item)
            
            cf_test_user_dict = collections.defaultdict(list)
            for user, item in cf_test_data:
                item = item + n_users # Make sure to remap items!!
                cf_test_user_dict[user].append(item)
            
            return ckg_data, ckg_head_dict, cf_train_user_dict, cf_test_user_dict

        # Get paths
        train_file, test_file, _, _, kg_file, _, _, pretrain_file = self.raw_paths

        # Load pretrained embeddings 
        user_item_pretrained = _load_pretrain(pretrain_file)

        # Load CF data (Bipartite Graph); Load CF User dict – Used for creating postive and negative samples
        cf_train_data = _load_cf_data(train_file)
        cf_test_data = _load_cf_data(test_file) 
        n_users = max(max(cf_train_data[:, 0]), max(cf_test_data[:, 0])) + 1 
        n_items = max(max(cf_train_data[:, 1]), max(cf_test_data[:, 1])) + 1 
        n_train_cf = np.int32(cf_train_data.shape[0])
        n_test_cf = np.int32(cf_test_data.shape[0])
  
        # Load KG
        kg_data = _load_kg(kg_file)
        n_entities = max(max(kg_data[:, 0]), max(kg_data[:, 2])) + 1
        n_relations = max(kg_data[:, 1]) + 1
        n_triples = np.int32(kg_data.shape[0])

        # Create CKG 
        ckg_data, ckg_head_dict, cf_train_user_dict, cf_test_user_dict = _create_ckg(cf_train_data, cf_test_data, kg_data, n_users, n_relations)
        n_nodes_ckg = n_users + n_entities
        n_relations_ckg = 2 * n_relations + 2
        n_triples_ckg = np.int32(len(ckg_data))

        # Get Edge Index und Edge Type – Sorted makes sure that everything is coalesced
        edge_list = sorted(ckg_data, key=lambda x: (x[0], x[1], x[2]))
        edge = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index, edge_type = edge[:2], edge[2] 

        print("Done!")

        print("Saving dataset...")
        data = Data()

        if self.model in ['kgat']:
            # Collaborative Knowledge Graph for Test and Evaluation (Sparse Adjacency Matrix)
            # Note: Only consists out of Train Users and Items, but the full KG
            data.edge_index = edge_index
            data.edge_type = edge_type

            # Remapped! User/Head Look-Ups (Adjacency Lists) for Positive / Negative Sampling during Training
            data.ckg_head_dict = ckg_head_dict
            data.cf_train_user_dict = cf_train_user_dict
            data.cf_test_user_dict = cf_test_user_dict

            # Pretrained Embeddings
            data.user_item_pretrained = user_item_pretrained

            # Some Helpers for For-Loops and Initializations
            data.n_users = n_users
            data.n_items = n_items
            data.n_nodes_ckg = n_nodes_ckg
            data.n_relations_ckg = n_relations_ckg

            # Some Helpers for Positive and Negative Sampling during Training
            data.n_train_cf = n_train_cf
            data.n_triples_ckg = n_triples_ckg

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}{}()'.format(self.name.upper(), self.__class__.__name__)