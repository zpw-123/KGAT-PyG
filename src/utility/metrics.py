import torch
import numpy as np

def precision_at_K_batch(hits, K):
    """ Calculate Precision@K
    Returns:
      res (ndarray): i-th value is the precision@K for user i (n_users_batch)
    """
    assert hits.shape[1] >= K
    assert K >= 1
    res = np.mean(hits[:, :K], axis=1)
    return res

def recall_at_K_batch(hits, K):
    """ Calculate Recall@K
    Returns:
      res (ndarray): i-th value is the recall@K for user i (n_users_batch)
    """
    assert hits.shape[1] >= K
    assert K >= 1
    res = np.sum(hits[:, :K], axis=1) / np.sum(hits, axis=1)
    return res

def ndcg_at_K_batch(hits, K):
    """ Calculate NDCG@K. Note that if hits is not binary, use
    np.sum((2 ** hits_K - 1) / np.log2(np.arange(2, K + 2)), axis=1).
    Note that the values produced by this function are correct. The authors of KGAT
    used heapq.nlargest(K_max, item_score, key=item_score.get) in batch_test.py, which
    cut off all ground truth items that were ranked below threshold K_max. 
    Thus the denominator idcg will not be correct for the KGAT paper.
    Returns:
      res (ndarray): i-th value is the ndcg@K for user i (n_users_batch)
    """
    assert hits.shape[1] >= K
    assert K >= 1
    hits_K = hits[:, :K]
    dcg = np.sum(hits_K / np.log2(np.arange(2, K + 2)), axis=1)

    sorted_hits_K = np.flip(np.sort(hits), axis=1)[:, :K]
    idcg = np.sum(sorted_hits_K / np.log2(np.arange(2, K + 2)), axis=1)
    idcg[idcg == 0] = np.inf

    res = (dcg / idcg)
    return res

def calc_metrics_at_k(pred_scores, train_user_dict, test_user_dict, user_batch, all_item_ids, Ks):

    # Create test_pos_item_binary matrix. 
    # v_ij is 1, if item j is a positive item for user i, 0 otherwise
    n_user = len(train_user_dict.keys())
    test_pos_item_binary = np.zeros([len(user_batch), len(all_item_ids)], dtype=np.float32)
    for i, user in enumerate(user_batch):
        train_pos_items = train_user_dict[user]
        test_pos_items = test_user_dict[user]
        #Redo remapping, so we can fill the matrix
        train_pos_items = [i - n_user for i in train_pos_items]
        test_pos_items = [i - n_user for i in test_pos_items]
        # Set pred_scores of train items to 0
        pred_scores[i][train_pos_items] = 0 
        test_pos_item_binary[i][test_pos_items] = 1

    # For each user-row: Order item indices based on their matching score (Eq. 12)
    try:
        # If cuda is available, speed uo sorting process
        _, rank_indices = torch.sort(pred_scores.cuda(), descending=True)
    except:
        _, rank_indices = torch.sort(pred_scores, descending=True)
    rank_indices = rank_indices.cpu()

    # Create hit marix, ordered by rank in descending order. 
    # v_ij is 1, if j-th ranked item is a positive item for user i ("hit"), 0 otherwise
    binary_hit = []
    for i in range(len(user_batch)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)

    precision_batch, recall_batch, ndcg_batch = [], [], []
    for K in Ks:
        precision_batch.append(precision_at_K_batch(binary_hit, K))
        recall_batch.append(recall_at_K_batch(binary_hit, K))
        ndcg_batch.append(ndcg_at_K_batch(binary_hit, K))

    return np.array(precision_batch), np.array(recall_batch), np.array(ndcg_batch) # (len(Ks), n_users_batch)