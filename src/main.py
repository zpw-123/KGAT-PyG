import torch
import numpy as np
import random
from time import time
from tqdm import tqdm

from datasets import KnowledgeAwareRecommendation
from transforms import KGATNorm

from models import KGAT
from models import KGATLinear

from utility.parser import parse_args
from utility.metrics import *

def train(args):
    model.train()

    #---------------------------Train CF---------------------------
    cf_start = time()
    cf_total_loss, cf_base_loss, cf_reg_loss = 0, 0, 0
    for batch_user, batch_pos_items, batch_neg_items in tqdm(cf_dataloader, position = 0):
        # Transfer to GPU
        batch_user, batch_pos_items, batch_neg_items = batch_user.to(device), batch_pos_items.to(device), batch_neg_items.to(device)

        # Perform Message Passing
        embeds_concat = model(data.edge_index)

        # Optimize 
        cf_batch_loss, cf_batch_base_loss, cf_batch_reg_loss = model.calc_cf_loss(embeds_concat, batch_user, batch_pos_items, batch_neg_items)
        cf_batch_loss.backward()
        cf_optimizer.step()
        cf_optimizer.zero_grad()

        # Save Loss
        cf_total_loss += cf_batch_loss.item()
        cf_base_loss += cf_batch_base_loss.item()
        cf_reg_loss += cf_batch_reg_loss.item()

    cf_time = time() - cf_start

    #---------------------------Train CKG---------------------------
    ckg_start = time()
    ckg_total_loss, ckg_base_loss, ckg_reg_loss = 0, 0, 0
    for batch_head, batch_relation, batch_pos_tail, batch_neg_tail in tqdm(ckg_dataloader, position = 0):
        # Transfer to GPU
        batch_head, batch_relation, batch_pos_tail, batch_neg_tail = batch_head.to(device), batch_relation.to(device), batch_pos_tail.to(device), batch_neg_tail.to(device)

        # Optimize
        ckg_batch_loss, ckg_batch_base_loss, ckg_batch_reg_loss = model.ckg_loss(batch_head, batch_relation, batch_pos_tail, batch_neg_tail)
        ckg_batch_loss.backward()
        ckg_optimizer.step()
        ckg_optimizer.zero_grad()

        # Save Loss
        ckg_total_loss += ckg_batch_loss.item()
        ckg_base_loss += ckg_batch_base_loss.item()
        ckg_reg_loss += ckg_batch_reg_loss.item()

    ckg_time = time() - ckg_start

    #---------------------------Attention---------------------------
    att_start = time()
    with torch.no_grad():
        attention = model.update_attention(data.edge_index, data.edge_type)

    att_time = time() - att_start
    total_time = att_time + cf_time + ckg_time
    
    if (epoch % args.verbose_steps) == 0:
        print('Training: Epoch {:04d} [{:.2f}s=={:.2f}s + {:.2f}s + {:.2f}s] | CF Loss [{:.4f}=={:.4f} + {:.4f}] | CKG Loss [{:.4f}=={:.4f} + {:.4f}]'.format(epoch, total_time, cf_time, ckg_time, att_time, cf_total_loss, cf_base_loss, cf_reg_loss, ckg_total_loss, ckg_base_loss, ckg_reg_loss))

@torch.no_grad()
def test(args):

    model.eval()

    #---------------------------Evaluation---------------------------
    eval_start = time()
    precision, recall, ndcg = [], [], []
    for user_batch in test_users_batches:

        # Transfer to gpu
        user_batch = user_batch.to(device)

        # Perform message passing
        embeds_concat = model(data.edge_index)

        # Get embeddings of users in batch and all items 
        user_embed = embeds_concat[user_batch]          # (test_batch_size, cf_concat_dim)
        item_embed = embeds_concat[all_item_ids]        # (n_items, cf_concat_dim)

        # Equation (12) - Predict Matching Score between Batch Test Users and Items
        pred_scores = torch.matmul(user_embed, item_embed.transpose(0, 1))   # (test_batch_size, n_eval_items)

        # Perform Evaluation
        pred_scores = pred_scores.cpu()
        user_batch = user_batch.cpu().numpy()
        all_items = all_item_ids.cpu().numpy()
        precision_batch, recall_batch, ndcg_batch = calc_metrics_at_k(pred_scores, data.cf_train_user_dict, data.cf_test_user_dict, user_batch, all_items, eval(args.Ks))

        precision.append(precision_batch)
        recall.append(recall_batch)
        ndcg.append(ndcg_batch)

    precision_k = np.sum(np.hstack(precision), axis=1) / len(test_users)
    recall_k = np.sum(np.hstack(recall), axis=1) / len(test_users)
    ndcg_k = np.sum(np.hstack(ndcg), axis=1) / len(test_users)

    eval_time = time() - eval_start

    print('Evaluation: Epoch {:04d} [{:.2f}s] | precision@K {} | recall@K {} | ndcg@K {}'.format(epoch, eval_time, precision_k, recall_k, ndcg_k))

    return precision_k, recall_k, ndcg_k


if __name__ == '__main__':

    # Get Args
    args = parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    dataset = KnowledgeAwareRecommendation(root='../data/', name=args.dataset, model=args.model)
    data = dataset[0]

    # Get Attention Prior
    sinorm = KGATNorm(adj_type='si')
    data = sinorm(data)
    data.to(device)

    # Model
    if args.model_type == 'linear':
        print("Using KGAT with Linear Layers.")
        model = KGATLinear(data, args)
    else:
        print("Using KGAT with Parameter/Bias Layers.")
        model = KGAT(data, args)
        
    if args.pretrain == 2:
        model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # Dataloader
    cf_dataloader = model.cf_loader(batch_size=args.cf_batch_size)
    ckg_dataloader = model.ckg_loader(batch_size=args.ckg_batch_size)

    # Optimizer
    cf_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckg_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Get test user batches
    val_batch_size = 2048
    test_users = list(data.cf_test_user_dict.keys())
    test_users_batches = [test_users[i:i+val_batch_size] for i in range(0, len(test_users), val_batch_size)]
    test_users_batches = [torch.LongTensor(d) for d in test_users_batches]
    test_users_batches = [d.to(device) for d in test_users_batches]
    
    # Get all remapped items for lookup during testing
    all_item_ids = torch.arange(data.n_items, dtype=torch.long) + data.n_users
    all_item_ids.to(device)

    # train model
    for epoch in range(0, args.epochs):

        train(args)

        if (epoch % args.evaluate_steps) == 0:
            precision_k, recall_k, ndcg_k = test(args)