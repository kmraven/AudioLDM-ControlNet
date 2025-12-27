import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConceptBranch(nn.Module):
    def __init__(self, out_dim, embedding_dim):
        super(ConceptBranch, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32, out_dim),
            nn.Softmax(dim=-1)  
        )

    def forward(self, x):  # x: (B*L, D)
        # input: [V1, V2] = (BL, D)
        # output: W = (BL, n_concepts)
        x = self.fc1(x)
        x = self.fc2(x)
        return x  # (B*L, num_concepts)


class ConditionalSimNet(nn.Module): 
    def __init__(self, n_conditions, embedding_size, learnedmask=True, prein=False):
        super(ConditionalSimNet, self).__init__()
        self.learnedmask = learnedmask
        self.num_conditions = n_conditions # number of parallel concepts
        self.embedding_size = embedding_size # size of V1 and masks

        # Condition-specific masks
        self.masks = nn.Embedding(n_conditions, embedding_size) # dictionary with vocab size=n_conditions, each vocab is a vector of embedding_size

        if learnedmask: # learnable masks
            if prein: # fix masks to be 90% masking out all except one part of the envenly splitted V1, learnable
                mask_array = np.full((n_conditions, embedding_size), 0.1, dtype=np.float32) # filling array of certain shape with 0.1
                # n_conditions evenly split C1 dim, each mask will leave one part as 1, the rest is 0.1.
                mask_len = embedding_size // n_conditions 
                for i in range(n_conditions): 
                    mask_array[i, i * mask_len:(i + 1) * mask_len] = 1.0 
                self.masks.weight.data.copy_(torch.tensor(mask_array))
            else: # all masks are randomly intialized to be around 0.9, learnable
                nn.init.normal_(self.masks.weight, mean=0.9, std=0.7)
        else: # fix masks to be masking out all except one part of the evenly splitted V1, not learnable 
            mask_array = np.zeros((n_conditions, embedding_size), dtype=np.float32)
            mask_len = embedding_size // n_conditions
            for i in range(n_conditions):
                mask_array[i, i * mask_len:(i + 1) * mask_len] = 1.0
            self.masks.weight.data.copy_(torch.tensor(mask_array))
            self.masks.weight.requires_grad = False

    def forward(self, x, c):
        """
        Args:
            x: [B*L, D] — framewise input
            c: [B*L] — condition indices for each sample

        Returns:
            masked_embedding: [B*L, D]
            mask_norm: scalar
            embed_norm: scalar
            masked_embed_norm: scalar
        """
        B_L, D = x.shape

        # Get per-sample masks: (B*L, D)
        mask = self.masks(c)  # [B*L, D] #c are indices, get BL masks, each mask is dim=D

        if self.learnedmask:
            mask = F.relu(mask)

        # Apply mask
        masked_embedding = x * mask  # [B*L, D]

        # Normalize masked embedding per feature
        norm = torch.norm(masked_embedding, p=2, dim=-1, keepdim=True) + 1e-10
        masked_embedding = masked_embedding / norm  # [B*L, D]

        # Regularization values, these are for monitoring
        mask_norm = torch.norm(mask.detach(), p=1) / B_L # average mask norm
        embed_norm = torch.norm(x, p=2) / B_L # average V1 norm
        masked_embed_norm = torch.norm(masked_embedding, p=2) / B_L # average V1*C1 norm

        return masked_embedding, mask_norm, embed_norm, masked_embed_norm
