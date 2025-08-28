from __future__ import print_function, absolute_import
import numpy as np

def evaluation(distmat, q_pids, g_pids, q_fldrs, g_fldrs, max_rank=20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        #print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)

    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):

        # get query pid and folders
        q_pid = q_pids[q_idx]
        q_fldr = q_fldrs[q_idx]

        # remove gallery samples that have the same pid and folders with query
        order = indices[q_idx]
        remove = (q_fldr == 3) & (g_fldrs[order] == 2)

        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        #print("matches",orig_cmc)
        #print("SUM",cmc)

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q  
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP
