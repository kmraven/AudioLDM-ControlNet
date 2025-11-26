import torch
import torch.nn.functional as F
import scipy.stats
import numpy as np
import concurrent.futures
import time


# Returns list of retrieved top k videos based on the sims matrix
def get_retrieved_videos(sims, k):
    argm = np.argsort(-sims, axis=1)
    topk = argm[:,:k].reshape(-1)
    retrieved_videos = np.unique(topk)
    return retrieved_videos

# Returns list of indices to normalize from sims based on videos
def get_index_to_normalize(sims, videos):
    argm = np.argsort(-sims, axis=1)[:,0]
    result = np.array(list(map(lambda x: x in videos, argm)))
    result = np.nonzero(result)
    return result

def qb_norm(train_test, test_test, k=1, beta=50):
    k = k
    beta = beta
    retrieved_videos = get_retrieved_videos(train_test, k)
    test_test_normalized = test_test
    train_test = np.exp(train_test*beta)
    test_test = np.exp(test_test*beta)

    normalizing_sum = np.sum(train_test, axis=0)
    index_for_normalizing = get_index_to_normalize(test_test, retrieved_videos)
    print(normalizing_sum.shape)
    print(len(index_for_normalizing))
    print(test_test_normalized.shape)
    test_test_normalized[index_for_normalizing, :] = \
        np.divide(test_test[index_for_normalizing, :], normalizing_sum)
    return test_test_normalized


def beat_align_score(gt, pred):
    if len(gt) == 0 or len(pred) == 0:
        return 0
    ba = 0
    for bb in gt:
        ba += np.exp(-np.min(((pred - bb))**2) / 2 / 9)
    return (ba / len(gt))

def comp(music_beat, motion_beats):
    similarity_matrix = np.apply_along_axis(comp1, 1, motion_beats, music_beat)
    return similarity_matrix

def comp1(motion_beat, music_beat):
    similarity = beat_align_score(music_beat, motion_beat)
    return similarity

def beat_similarity(music_beats, motion_beats):
    similarity_matrix = np.apply_along_axis(comp, 1, music_beats, motion_beats)
    return similarity_matrix


def sim_matrix_training_per_frame(text_embeds, vid_embeds):
    """
    Computes cosine similarity between all frame pairs.

    Args:
        text_embeds: (B1, L, D)
        vid_embeds:  (B2, L, D)

    Returns:
        sims: (B1*L, B2*L)
    """
    B, L, D = text_embeds.shape
    
    # Normalize along feature dimension
    text_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # (B, L, D)
    vid_norm = vid_embeds / vid_embeds.norm(dim=-1, keepdim=True)    # (B, L, D)

    # Reshape for broadcasting: (B, 1, L, D) vs (1, B, L, D)
    text_norm = text_norm.unsqueeze(1)  # (B, 1, L, D)
    vid_norm = vid_norm.unsqueeze(0)    # (1, B, L, D)

    # Compute dot product per frame, then sum over features and frames
    sims = (text_norm * vid_norm).sum(dim=-1).sum(dim=-1)  # (B, B)

    return sims



def sim_matrix_inference_per_frame(text_embeds_per_video_id, vid_embeds_per_video_id):

    """
    Computes per-frame cosine similarity between all text queries and all videos (frame-aligned).

    Args:
        text_embeds_per_video_id: (V_text, T, L, D), embeddings for all query clips 
        vid_embeds_per_video_id:  (V_vid, L, D) , embeddings for all v

    Returns:
        sims: (V_text, T, V_vid, L)
    """

    V_text, T, L, D = text_embeds_per_video_id.shape # T is 1 n this case (V_text = total number of all query clips)
    V_vid, L2, D2 = vid_embeds_per_video_id.shape  # (V_vid = total number of all videos)

    assert L == L2 and D == D2, f"Shape mismatch: text (L={L}, D={D}), video (L={L2}, D={D2})"

    # Normalize 
    # L2 Norm along D, Cosine similarity is dot product of L2 normalized vectors
    text_norm = text_embeds_per_video_id / text_embeds_per_video_id.norm(dim=-1, keepdim=True) # (V_text, T, L, D)
    video_norm = vid_embeds_per_video_id / vid_embeds_per_video_id.norm(dim=-1, keepdim=True) # (V_vid, L, D)

    # Einstein summation 
    # atld => Norma,ized Music Embeddings
    # bld = Normalized Video Embeddings
    # atbl = resulting similarity matrix 
    sims = torch.einsum('atld,bld->atbl', text_norm, video_norm)  # (V_text, T, V_vid, L) compute similarity of each pair of frame

    # Aggregate across frames
    sims = sims.sum(dim=-1) 

    # Each sims[i][j] represents total alignment score between text clip i and video j
    
    return sims


def sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type='avg'):
    """
    Computes the similarity matrix using pooled video frames

    Output
        sims: num_texts x num_vids
    """
    #import pdb
    #pdb.set_trace()
    B, T, D  = text_embeds.shape
    text_embeds = text_embeds.reshape((B, T*D))
    vid_embeds_pooled = vid_embeds_pooled.reshape((B, T*D))

    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True) #(B,LD)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())

    else:
        # num_texts x embed_dim x num_vids
        vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)
        # num_texts x 1 x embed_dim
        text_embeds = text_embeds.unsqueeze(1)

        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)

    return sims

def sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, pooling_type='avg'):
    """
    Computes the similarity matrix using pooled video frames using all texts per video

    Output
        sims: num_vids x max_text_per_vid x num_vids
    """
    B, _, T, D  = text_embeds_per_video_id.shape
    text_embeds_per_video_id = text_embeds_per_video_id.reshape((B, 1, T*D))
    vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.reshape((B, T*D))
    text_embeds_per_video_id = text_embeds_per_video_id / text_embeds_per_video_id.norm(dim=-1, keepdim=True) #[B, 1, L *D]
    vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id / vid_embeds_pooled_per_video_id.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        # text_embeds_per_video_id -> num_vids x max_text_per_vid x embed_dim
        # vid_embeds_pooled_per_video_id -> num_vids x embed_dim

        sims = text_embeds_per_video_id @ vid_embeds_pooled_per_video_id.t()

    else:
        # text_embeds_per_video_id -> num_vids x max_text_per_vid x embed_dim (B,1,D) there is no L
        # vid_embeds_pooled_per_video_id -> num_vids x num_vids x max_text_per_vid x embed_dim # (B,B,1,D)
        num_vids, max_text_per_vid, embed_dim = text_embeds_per_video_id.shape

        # num_vids x max_text_per_vid x embed_dim x num_vids
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.permute(1,2,3,0)
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.view(num_vids*max_text_per_vid, embed_dim, num_vids) #(B,D,B)
        # num_vids x max_text_per_vid x 1 x embed_dim
        text_embeds_per_video_id = text_embeds_per_video_id.unsqueeze(2)
        text_embeds_per_video_id = text_embeds_per_video_id.view(num_vids*max_text_per_vid, 1, embed_dim) #(B,1,D)

        sims = torch.bmm(text_embeds_per_video_id, vid_embeds_pooled_per_video_id) #(B,1,B)
        sims = sims.view(num_vids, max_text_per_vid, 1, num_vids).squeeze(2)

    return sims

def sim_matrix_training_per_frame_aspect_max(text_embeds, vid_embeds):
    """
    Computes similarity between each (text, video) pair by max over aspects per frame.

    Args:
        text_embeds: (B, L, K, D)
        vid_embeds:  (B, L, K, D)

    Returns:
        sims: (B, B) similarity matrix
    """
    B, L, K, D = text_embeds.shape

    # Normalize
    text_norm = F.normalize(text_embeds, dim=-1)  # (B, L, K, D)
    vid_norm = F.normalize(vid_embeds, dim=-1)    # (B, L, K, D)

    # Reshape for broadcasting
    text_norm = text_norm.unsqueeze(1)  # (B, 1, L, K, D)
    vid_norm = vid_norm.unsqueeze(0)   # (1, B, L, K, D)

    # Compute cosine similarity over last dim → (B, B, L, K)
    sim = (text_norm * vid_norm).sum(dim=-1)

    # Take max over aspect dimension (K) → (B, B, L)
    sim_max = sim.max(dim=-1).values

    # Sum over time (L) → (B, B)
    sims = sim_max.sum(dim=-1)

    return sims


def sim_matrix_inference_per_frame_aspect_max(text_embeds_per_video_id, vid_embeds_per_video_id):
    """
    Computes per-frame per-aspect max similarity between text sequences and videos.

    Args:
        text_embeds_per_video_id: (V_text, T, L, K, D)
        vid_embeds_per_video_id:  (V_vid, L, K, D)

    Returns:
        sims: (V_text, T, V_vid)
    """
    V_text, T, L, K, D = text_embeds_per_video_id.shape
    V_vid, L2, K2, D2 = vid_embeds_per_video_id.shape

    assert L == L2 and K == K2 and D == D2, "Shape mismatch"

    # Normalize
    text_norm = text_embeds_per_video_id / text_embeds_per_video_id.norm(dim=-1, keepdim=True)  # (V_text, T, L, K, D)
    video_norm = vid_embeds_per_video_id / vid_embeds_per_video_id.norm(dim=-1, keepdim=True)  # (V_vid, L, K, D)
    #import pdb
    #pdb.set_trace()
    # Compute dot product per frame/aspect: einsum → (V_text, T, V_vid, L, K)
    sims = torch.einsum('atlkd,blkd->atblk', text_norm, video_norm)

    # Max over K → (V_text, T, V_vid, L)
    sims_max = sims.max(dim=-1).values

    # Sum (or mean) over frames → (V_text, T, V_vid)
    sims_out = sims_max.sum(dim=-1)

    return sims_out


def generate_embeds_per_video_id(text_embeds, vid_embeds_pooled, all_vid_ids, pooling_type):
    # Construct dictionary of text embeds per unique video id
    text_embeds_per_video_id = {}

    for idx, v_id in enumerate(all_vid_ids):
        if v_id in text_embeds_per_video_id:
            text_embeds_per_video_id[v_id].append(text_embeds[idx])
        else:
            text_embeds_per_video_id[v_id] = [text_embeds[idx]]

    for v_id in text_embeds_per_video_id:
        text_embeds_per_video_id[v_id] = torch.stack(text_embeds_per_video_id[v_id])

    # num_vids x max_text_per_vid x embed_dim
    text_embeds_per_video_id = pad_and_stack_dict_to_tensor(text_embeds_per_video_id,
        text_embeds_per_video_id.keys(), text_embeds.shape[-1])

    if pooling_type == 'avg':
        # num_vids x embed_dim
        vid_embeds_pooled_per_video_id = vid_embeds_pooled

    else:
        # Construct dictionary of video embeds for each text per video_id
        vid_embeds_pooled_per_video_id = []

        for i in range(vid_embeds_pooled.shape[0]):
            vid_embeds_pooled_per_video_id.append({})
            for idx, v_id in enumerate(all_vid_ids):
                if v_id in vid_embeds_pooled_per_video_id[i]:
                    vid_embeds_pooled_per_video_id[i][v_id].append(vid_embeds_pooled[i, idx, :])
                else:
                    vid_embeds_pooled_per_video_id[i][v_id] = [vid_embeds_pooled[i, idx, :]]

        for i in range(len(vid_embeds_pooled_per_video_id)):
            for v_id in vid_embeds_pooled_per_video_id[i]:
                vid_embeds_pooled_per_video_id[i][v_id] = torch.stack(vid_embeds_pooled_per_video_id[i][v_id])

            # num_vids x max_text_per_vid x embed_dim
            vid_embeds_pooled_per_video_id[i] = pad_and_stack_dict_to_tensor(vid_embeds_pooled_per_video_id[i],
                    vid_embeds_pooled_per_video_id[i].keys(), vid_embeds_pooled.shape[-1])

        # num_vids x num_vids x max_text_per_vid x embed_dim
        vid_embeds_pooled_per_video_id = torch.stack(vid_embeds_pooled_per_video_id)

    return text_embeds_per_video_id, vid_embeds_pooled_per_video_id


def t2v_metrics(sims): #sims: (N_audio, 1, N_video)
    # Permute sims so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    print("sims dimension", sims.shape)
    stacked_sims = sims.permute(1,0,2) # (1, N_audio, N_video)

    sims_sort = torch.argsort(stacked_sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.flatten(torch.diagonal(sims_sort_2, dim1=1, dim2=2)) # ranking of the similarity, max rank determined by # of samples in validation set

    # Now we need to extract valid ranks, as some belong to inf padding values
    valid_check = torch.flatten(torch.diagonal(sims, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(valid_check), torch.isnan(valid_check))
    valid_ranks = ranks[mask]


    return compute_metrics(valid_ranks.numpy())


def v2t_metrics(sims):
    # Code to avoid nans
    sims[sims!=sims] = float('-inf')
    # Forms a similarity matrix
    sims, _ = torch.max(sims, dim = 1)
    sims = sims.t()

    sims_sort = torch.argsort(sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.diag(sims_sort_2).numpy() # diagonal

    return compute_metrics(ranks)


def compute_metrics(lst):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(lst == 0)) / len(lst)
    metrics["R5"] = 100 * float(np.sum(lst < 5)) / len(lst)
    metrics["R10"] = 100 * float(np.sum(lst < 10)) / len(lst)
    metrics["R50"] = 100 * float(np.sum(lst < 50)) / len(lst)
    metrics["R100"] = 100 * float(np.sum(lst < 100)) / len(lst)
    metrics["MedR"] = np.median(lst) + 1
    metrics["MeanR"] = np.mean(lst) + 1
    metrics["AVG"] = (metrics["R1"] + metrics["R5"] + metrics["R10"] + metrics["R50"] + metrics["R100"] - metrics["MedR"] / 10 - metrics["MeanR"] / 10)
    #stats = [metrics[x] for x in ("R1", "R5", "R10")]
    #metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics


def compute_directory_based_metrics(sims, music_ids, video_ids):
    """
    Compute metrics where all videos from the same music directory are considered correct.

    Args:
        sims: similarity matrix (N_music, 1, N_video)
        music_ids: list of music IDs
        video_ids: list of video IDs

    Returns:
        metrics: dictionary containing directory-based metrics
    """
    sims_squeezed = sims.squeeze(1)  # (N_music, N_video)

    # Extract music directory prefix (e.g., "mWA0" from "mWA0/gWA_sBM_c01_d25_mWA0_ch02_1")
    def get_music_dir(video_id):
        return video_id.split('/')[0]

    music_dirs = [get_music_dir(vid_id) for vid_id in music_ids]
    video_dirs = [get_music_dir(vid_id) for vid_id in video_ids]

    avg_ranks = []
    all_correct_ranks = []  # Store all individual ranks for correct videos

    for i in range(len(sims_squeezed)):
        query_music_dir = music_dirs[i]

        # Get sorted indices by similarity (descending)
        sorted_indices = torch.argsort(sims_squeezed[i], descending=True).cpu().numpy()

        # Find ranks of all videos from the same music directory
        correct_ranks = []
        for rank, idx in enumerate(sorted_indices):
            if video_dirs[idx] == query_music_dir:
                correct_ranks.append(rank + 1)  # rank is 1-indexed

        if len(correct_ranks) > 0:
            avg_rank = np.mean(correct_ranks)
            avg_ranks.append(avg_rank)
            all_correct_ranks.extend(correct_ranks)

    avg_ranks = np.array(avg_ranks)
    all_correct_ranks = np.array(all_correct_ranks)

    # Compute metrics based on average ranks per query
    metrics = {}
    metrics["Dir_R1"] = 100 * float(np.sum(avg_ranks <= 1)) / len(avg_ranks)
    metrics["Dir_R5"] = 100 * float(np.sum(avg_ranks <= 5)) / len(avg_ranks)
    metrics["Dir_R10"] = 100 * float(np.sum(avg_ranks <= 10)) / len(avg_ranks)
    metrics["Dir_R50"] = 100 * float(np.sum(avg_ranks <= 50)) / len(avg_ranks)
    metrics["Dir_MedR"] = np.median(avg_ranks)
    metrics["Dir_MeanR"] = np.mean(avg_ranks)

    # Also compute metrics on all individual correct video ranks
    metrics["All_R1"] = 100 * float(np.sum(all_correct_ranks == 1)) / len(all_correct_ranks) if len(all_correct_ranks) > 0 else 0
    metrics["All_R5"] = 100 * float(np.sum(all_correct_ranks <= 5)) / len(all_correct_ranks) if len(all_correct_ranks) > 0 else 0
    metrics["All_R10"] = 100 * float(np.sum(all_correct_ranks <= 10)) / len(all_correct_ranks) if len(all_correct_ranks) > 0 else 0
    metrics["All_MedR"] = np.median(all_correct_ranks) if len(all_correct_ranks) > 0 else 0
    metrics["All_MeanR"] = np.mean(all_correct_ranks) if len(all_correct_ranks) > 0 else 0

    return metrics, avg_ranks, all_correct_ranks


def pad_and_stack_dict_to_tensor(input, order, d=512):
    max_length = max([input[k].shape[0] for k in input])

    padded_input = {k: torch.cat([input[k], torch.full((max_length - input[k].shape[0], d),
                                                        float("-inf"), device = input[k].device)]) for k in input}

    padded_stacked_input = torch.stack([padded_input[k] for k in order], dim = 0)
    return padded_stacked_input


