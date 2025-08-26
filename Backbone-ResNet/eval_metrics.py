from __future__ import print_function, absolute_import
from PIL import Image
import numpy as np
import os
import shutil


def evaluation(distmat, q_pids, g_pids, q_folders, g_folders, q_paths=None, g_paths=None, max_rank=20, retrieval_imgs=False, retrieval_dir='retrieval_folder'):

    if retrieval_imgs:
        if os.path.exists(retrieval_dir):
            shutil.rmtree(retrieval_dir)
        os.makedirs(retrieval_dir)

    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    
    num_valid_q = 0. # number of valid query
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_folder = q_folders[q_idx]

        # remove gallery samples that have the same pid and folderid with query
        order = indices[q_idx]
        remove = (q_folder == 3) & (g_folders[order] == 2)
        keep = np.invert(remove)
        
        # compute cmc curve    
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches

        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        
        if retrieval_imgs:
            query_img = Image.open(q_paths[q_idx])
            query_pid = q_pid

            # Create directory per person ID and per query index
            vis_q_dir = os.path.join(retrieval_dir, f'pid_{query_pid}', f'q_{q_idx}')
            os.makedirs(vis_q_dir, exist_ok=True)

            # Save the query image
            query_img.save(os.path.join(vis_q_dir, f'query_pid{query_pid}.jpg'))

            topk = min(max_rank, len(order))
            rank_idx = 0

            for i in range(len(order)):
                if not keep[i]:
                    continue
                gallery_index = order[i]
                gallery_img = Image.open(g_paths[gallery_index])
                gallery_pid = g_pids[gallery_index]
                gall_folder = g_folders[gallery_index]
                correct = 'match' if gallery_pid == query_pid else 'nonmatch'

                # Save the query image
                gallery_img.save(os.path.join(vis_q_dir, f'rank{rank_idx+1}_{correct}_pid{gallery_pid}_cam{gall_folder}.jpg'))

                rank_idx += 1
                if rank_idx >= max_rank:
                    break
        
        # compute cummulative sum
        cmc = orig_cmc.cumsum()
        #print("CMC Matches",orig_cmc)
        #print("Cummulative Sum",cmc)

        # compute mINP
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q  
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP
    
    