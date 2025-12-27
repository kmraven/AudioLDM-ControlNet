################################## Trainer for Frame-wise Max-Concpet Learning w/ LSC ##################################
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict, deque
from config.base_config import Config
from trainer.base_trainer import BaseTrainer
from modules.metrics import (
    sim_matrix_training_per_frame_aspect_max,
    sim_matrix_training_per_frame,
    sim_matrix_inference_per_frame_aspect_max,
    sim_matrix_inference,
    generate_embeds_per_video_id,
    beat_similarity,
    qb_norm
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random


class Trainer(BaseTrainer):
    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader,
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None,
                 qb_norm=None, num_concepts=None, concept_branch=None, conditional_sim=None):
        super().__init__(model, loss, metrics, optimizer, config, writer)

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.tokenizer = tokenizer
        self.lr_scheduler = lr_scheduler

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -400.0
        self.best = -400.0

        self.qb_norm = qb_norm
        self.qbnorm_beta = config.qbnorm_beta
        self.qbnorm_k = config.qbnorm_k
        self.metric = config.metric

        self.num_concepts = num_concepts
        self.concept_branch = concept_branch.to(self.device)
        self.conditional_sim = conditional_sim.to(self.device)
        self.embed_loss_weight = 5e-3
        self.mask_loss_weight = 5e-4

        self.enable_probing = config.enable_probing
        self.save_inference = config.save_inference
        self.output_dir = config.output_dir
        self.exp_name = config.exp_name


    def save_results(self, sims, test_data_loader, output_dir, exp_name):
        """Save Top-10 inference results to a file."""
        if self.qb_norm is not None:
            result_file = os.path.join(output_dir, "inference_" + exp_name + "_top10_qbnorm.txt")
        else:
            result_file = os.path.join(output_dir, "inference_" + exp_name + "_top10.txt")
    
        # Get the filenames (used as both queries and candidates here)
        all_filenames = test_data_loader.dataset.data

        # Log shapes for sanity
        print(f"[INFO] Similarity matrix shape: {sims.shape}")
        print(f"[INFO] Number of filenames: {len(all_filenames)}")

        with open(result_file, "w") as f:
            for i in range(sims.shape[0]):
                query_name = all_filenames[i] 

                # Get Top-10 ranked indices (descending similarity)
                top_indices = torch.argsort(sims[i], descending=True).flatten()[:10]

                # Map to filenames and write
                f.write(f"{query_name}:\n")
                for rank, idx in enumerate(top_indices, 1):
                    candidate_name = all_filenames[idx.item()]
                    f.write(f"  Top-{rank}: {candidate_name}\n")
                f.write("\n")

        print(f"[INFO] Top-10 inference results saved to {result_file}")


    def _compute_concept_embedding(self, music_embed, video_embed):
        """
        Args:
            music_embed: (B, L, D)
            video_embed: (B, L, D)
        Returns:
            sim_loss: scalar similarity loss
            embed_norm: regularization term
            mask_norm: regularization term
            embedded_music: (B, L, K, D')
            embedded_video: (B, L, K, D')
        """
        B, L, D = music_embed.shape
        # normalize music and dance features by feature dimension
        music_feat = music_embed / (torch.norm(music_embed, p=2, dim=-1, keepdim=True) + 1e-10)
        video_feat = video_embed / (torch.norm(video_embed, p=2, dim=-1, keepdim=True) + 1e-10)
        # compute concept weights with input as concatenated music and dance features
        concept_input = torch.cat([music_feat, video_feat], dim=-1).view(B * L, -1) # (BL,2D)
        concept_weights = self.concept_branch(concept_input).view(B, L, self.num_concepts) # one value per concept per frame

        music_flat = music_embed.view(B * L, D)
        video_flat = video_embed.view(B * L, D)

        music_per_concept = []
        video_per_concept = []
        mask_norm = 0.0
        embed_norm = 0.0

        for i in range(self.num_concepts):
            concept_idx = torch.full((B * L,), i, dtype=torch.long, device=self.device)

            m_emb, m_mask, m_norm, _ = self.conditional_sim(music_flat, concept_idx) #(BL, D) apply masks on each frame
            v_emb, v_mask, v_norm, _ = self.conditional_sim(video_flat, concept_idx)

            if mask_norm is None:
                mask_norm = v_mask
            else:
                mask_norm = mask_norm + v_mask

            weight = concept_weights[:, :, i].reshape(B * L, 1)
            m_weighted = m_emb * weight # (B*L, D)
            v_weighted = v_emb * weight # (B*L, D)

            m_weighted_reshape = m_weighted.view(B, L, -1) # (B, L, D)
            v_weighted_reshape = v_weighted.view(B, L, -1) # (B, L, D)

            music_per_concept.append(m_weighted_reshape.unsqueeze(2))  # (B, L, 1, D)
            video_per_concept.append(v_weighted_reshape.unsqueeze(2))  # (B, L, 1, D)

        embedded_music = torch.cat(music_per_concept, dim=2) # (B, L, K, D)
        embedded_video = torch.cat(video_per_concept, dim=2) # (B, L, K, D)

        sim = sim_matrix_training_per_frame_aspect_max(embedded_music, embedded_video)
        sim_loss = self.loss(sim, self.model.clip_logit_scale) # clip_logit_scale is a scalar 4.6

        mask_norm /= self.num_concepts
        embed_norm = (m_norm + v_norm) / 2  

        return sim_loss, embed_norm, mask_norm, embedded_music, embedded_video

    def _compute_losses(self, music_embed, video_embed, beat_music, beat_video):
        sim_loss, embed_norm, mask_norm, fused_music, fused_video = self._compute_concept_embedding(music_embed, video_embed)

        # beat_sim = sim_matrix_training_per_frame(beat_music, beat_video)
        # beat_loss = self.loss(beat_sim, self.model.clip_logit_scale)

        B, L, _ = music_embed.shape
        loss = sim_loss #+ self.beta * beat_loss
        loss += self.embed_loss_weight * (embed_norm / np.sqrt(B * L))
        loss += self.mask_loss_weight * (mask_norm / (B * L))

        return loss, fused_music, fused_video


    def _train_epoch(self, epoch, model_len):
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps - 1, self.evals_per_epoch + 1, dtype=int)[1:]

        for batch_idx, data in enumerate(self.train_data_loader):
            # music: [32, 10, 768], video: [32, 10, 512]
            # music_beat: [32, 10, 100], video_beat: [32, 10, 100]
            for key in ['music', 'pose', 'music_beat', 'video_beat']:
                data[key] = data[key].to(self.device)

            # music embed: [32, 10, 256] video embed: [32, 10, 256]
            music_embed, video_embed = self.model(data, 'train')

            # BeatDance Only
            # output = sim_matrix_training_per_frame(music_embed['music_fuse'], video_embed['video_fuse'])
            # output1 = sim_matrix_training_per_frame(music_embed['music_beat'], video_embed['video_beat'])
            # loss = self.loss(output, self.model.clip_logit_scale) + self.beta * self.loss(output1, self.model.clip_logit_scale)

            loss, _, _ = self._compute_losses(
                music_embed['music_fuse'], video_embed['video_fuse'],
                music_embed['music_beat'], video_embed['video_beat']
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_scheduler:
                self.lr_scheduler.step()

            torch.clamp_(self.model.clip_logit_scale.data, max=np.log(100))

            self.global_step += 1
            if self.writer:
                self.writer.add_scalar('train/loss_train', loss.item(), self.global_step)

            total_loss += loss.detach().item()
            
            if batch_idx % self.log_step == 0:
                print(f"Train Epoch: {epoch} dl: {batch_idx}/{num_steps-1} Loss: {loss.detach().item():.6f}")

            if batch_idx in eval_steps:
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps - 1, model_len, model_len)
                self.model.train()

                if val_res['AVG-window'] > self.best_window:
                    self.best_window = val_res['AVG-window']
                    self._save_checkpoint(epoch, save_best=True)
                if val_res['AVG'] > self.best:
                    self.best = val_res['AVG']
                print(f" Current Best Window Average R@1 is {self.best_window}")
                print(f" Current Best R@1 is {self.best}\n")

        return {'loss_train': total_loss / num_steps}


    def _valid_epoch_step(self, epoch, step, num_steps, model_len, eval_len):
        self.model.eval()
        total_val_loss = 0.0
        fused_music_arr, fused_video_arr = [], []

        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                for key in ['music', 'pose', 'music_beat', 'video_beat']:
                    data[key] = data[key].to(self.device)

                music_embed, video_embed = self.model(data)

                loss, fused_music, fused_video = self._compute_losses(
                    music_embed['music_fuse'], video_embed['video_fuse'],
                    music_embed['music_beat'], video_embed['video_beat']
                )

                total_val_loss += loss.item()
                fused_music_arr.append(fused_music)   # [B, L, K, D]
                fused_video_arr.append(fused_video)   # [B, L, K, D]


        # Combine all batches: [B, L, K, D]
        fused_music = torch.cat(fused_music_arr, dim=0)
        fused_video = torch.cat(fused_video_arr, dim=0)

        sims = sim_matrix_inference_per_frame_aspect_max(fused_music.unsqueeze(1), fused_video).cpu().detach() # sims shape: [V_text, T, V_vid]
        if self.save_inference:
            self.save_results(sims, self.valid_data_loader, self.output_dir, self.exp_name)
    
        total_val_loss /= len(self.valid_data_loader)
        res = self.metrics(sims)

        for m in res:
            self.window_metric[m].append(res[m])

        for m in self.window_metric:
            res[m + "-window"] = np.mean(self.window_metric[m])

        print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----")
        for m in ['R1', 'R5', 'R10', 'R50', 'R100', 'MedR', 'MeanR', 'AVG']:
            print(f"{m}: {res[m]:.2f} (window: {res[m + '-window']:.2f})")
        print(f"Loss: {total_val_loss:.4f}")

        res['loss_val'] = total_val_loss

        if self.writer:
            for m in res:
                self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

        return res
        

    def _valid_qbnorm_epoch_step(self, epoch, step, num_steps, model_len, eval_len):
        self.model.eval()
        total_val_loss = 0.0
        fused_music_arr, fused_video_arr = [], []

        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                for key in ['music', 'pose', 'music_beat', 'video_beat']:
                    data[key] = data[key].to(self.device)

                music_embed, video_embed = self.model(data)

                loss, fused_music, fused_video = self._compute_losses(
                    music_embed['music_fuse'], video_embed['video_fuse'],
                    music_embed['music_beat'], video_embed['video_beat']
                )

                total_val_loss += loss.item()
                fused_music_arr.append(fused_music)   # [B, L, K, D]
                fused_video_arr.append(fused_video)   # [B, L, K, D]

                # #BeatDance Only
                # output = sim_matrix_training_per_frame(music_embed['music_fuse'], video_embed['video_fuse'])
                # output1 = sim_matrix_training_per_frame(music_embed['music_beat'], video_embed['video_beat'])
                # loss = self.loss(output, self.model.clip_logit_scale) + self.beta * self.loss(output1, self.model.clip_logit_scale)
                # total_val_loss += loss.item()
                # fused_music_arr.append(music_embed['music_fuse'])   
                # fused_video_arr.append(video_embed['video_fuse'])  

        # Combine all batches: [B, L, K, D]
        fused_music = torch.cat(fused_music_arr, dim=0)
        fused_video = torch.cat(fused_video_arr, dim=0)

        def _get_train_embeds(direction):
            embed_list = []

            for _, data in tqdm(enumerate(self.train_data_loader)):
                for key in ['music', 'pose', 'music_beat', 'video_beat']:
                    data[key] = data[key].to(self.device)

                music_out, video_out = self.model(data)

                if direction == 'v2t':
                    _, _, _, _, fused_video = self._compute_concept_embedding(
                        music_out['music_fuse'], video_out['video_fuse']
                    )
                    embed_list.append(fused_video)  # [B, L, K, D]
                else:
                    _, _, _, fused_music, _ = self._compute_concept_embedding(
                        music_out['music_fuse'], video_out['video_fuse']
                    )
                    embed_list.append(fused_music)  # [B, L, K, D]

            return torch.cat(embed_list, dim=0)  # [V_train, L, K, D]

        def _apply_qbnorm(train_sims, test_sims, transpose=False):
            train_sims = np.array(train_sims).squeeze(1)
            test_sims = np.array(test_sims).squeeze(1)

            if transpose:
                test_sims = test_sims.T

            test_sims = qb_norm(train_sims, test_sims, self.qbnorm_k, self.qbnorm_beta)

            if transpose:
                test_sims = test_sims.T

            return torch.from_numpy(test_sims).unsqueeze(1)

        if self.qb_norm is not None:
            if self.metric == 'v2t' or self.metric == 'v2t_frame':
                test_embed_sims = sim_matrix_inference_per_frame_aspect_max(fused_video.unsqueeze(1), fused_music).cpu().detach()
                train_embed = _get_train_embeds('v2t')
                train_embed_sims = sim_matrix_inference_per_frame_aspect_max(train_embed.unsqueeze(1), fused_music).cpu().detach()
                sims = _apply_qbnorm(train_embed_sims, test_embed_sims, transpose=True)
            else:
                test_embed_sims = sim_matrix_inference_per_frame_aspect_max(fused_music.unsqueeze(1), fused_video).cpu().detach()
                train_embed = _get_train_embeds('t2v')
                train_embed_sims = sim_matrix_inference_per_frame_aspect_max(train_embed.unsqueeze(1), fused_video).cpu().detach()
                sims = _apply_qbnorm(train_embed_sims, test_embed_sims, transpose=False)

        if self.save_inference:
            self.save_results(sims, self.valid_data_loader, self.output_dir, self.exp_name)

        total_val_loss /= len(self.valid_data_loader)
        res = self.metrics(sims)

        for m in res:
            self.window_metric[m].append(res[m])

        for m in self.window_metric:
            res[m + "-window"] = np.mean(self.window_metric[m])

        print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----")
        for m in ['R1', 'R5', 'R10', 'R50', 'R100', 'MedR', 'MeanR', 'AVG']:
            print(f"{m}: {res[m]:.2f} (window: {res[m + '-window']:.2f})")
        print(f"Loss: {total_val_loss:.4f}")

        res['loss_val'] = total_val_loss

        if self.writer:
            for m in res:
                self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

        return res

    
    def _get_concept_weights(self, music_embed, video_embed):
        """
        Computes normalized concept weights from music and video embeddings.
        Args:
            music_embed: (B, L, D)
            video_embed: (B, L, D)
        Returns:
            concept_weights: (B, L, K)
        """
        music_feat = music_embed / (torch.norm(music_embed, p=2, dim=-1, keepdim=True) + 1e-10)
        video_feat = video_embed / (torch.norm(video_embed, p=2, dim=-1, keepdim=True) + 1e-10)
        B, L, _ = music_feat.shape
        concept_input = torch.cat([music_feat, video_feat], dim=-1).view(B * L, -1)
        concept_weights = self.concept_branch(concept_input).view(B, L, self.num_concepts)
        return concept_weights


    def _apply_concept_weights(self, music_embed, video_embed, concept_weights):
        """
        Applies given concept weights to embeddings using conditional_sim.
        Args:
            music_embed: (B, L, D)
            video_embed: (B, L, D)
            concept_weights: (B, L, K)
        Returns:
            embedded_music: (B, L, K, D)
            embedded_video: (B, L, K, D)
        """
        B, L, D = music_embed.shape
        music_flat = music_embed.view(B * L, D)
        video_flat = video_embed.view(B * L, D)

        music_per_concept = []
        video_per_concept = []

        for k in range(self.num_concepts):
            concept_idx = torch.full((B * L,), k, dtype=torch.long, device=self.device)
            m_emb, _, _, _ = self.conditional_sim(music_flat, concept_idx)
            v_emb, _, _, _ = self.conditional_sim(video_flat, concept_idx)

            weight = concept_weights[:, :, k].reshape(B * L, 1)
            m_weighted = m_emb * weight
            v_weighted = v_emb * weight

            music_per_concept.append(m_weighted.view(B, L, 1, D))
            video_per_concept.append(v_weighted.view(B, L, 1, D))

        embedded_music = torch.cat(music_per_concept, dim=2)  # (B, L, K, D)
        embedded_video = torch.cat(video_per_concept, dim=2)  # (B, L, K, D)
        return embedded_music, embedded_video


    def probe_embedding_variants(self, num_samples=10):
        """
        Probes dance-side and music-side embeddings under different concept weights,
        and visualizes PCA + t-SNE for both.
        """
        self.model.eval()
        all_dance_embeddings, all_music_embeddings = [], []
        dance_labels, music_labels = [], []

        collected = []
        with torch.no_grad():
            for data in self.valid_data_loader:
                for key in ['music', 'pose', 'music_beat', 'video_beat']:
                    data[key] = data[key].to(self.device)
                music_embed, video_embed = self.model(data)
                collected.append((music_embed['music_fuse'], video_embed['video_fuse']))
                if len(collected) >= num_samples + 1:
                    break

            for i in range(num_samples):
                d_i, m_i = collected[i]
                d_j, m_j = collected[(i + 1) % len(collected)]
                j = (i + 1) % len(collected)

                # Concept weights
                w_ii = self._get_concept_weights(d_i, m_i)
                w_ij = self._get_concept_weights(d_i, m_j)
                w_jj = self._get_concept_weights(d_j, m_j)
                w_ji = self._get_concept_weights(d_j, m_i)

                # Dance embeddings
                for tag, w in [
                    (f"d({i},{i},{i},{i})", w_ii),
                    (f"d({i},{i},{i},{j})", w_ij),
                    (f"d({i},{i},{j},{j})", w_jj),
                ]:
                    fused_d, _ = self._apply_concept_weights(d_i, m_i, w)
                    emb = fused_d.max(dim=2).values.sum(dim=1).cpu().numpy() ####
                    # emb = fused_d.mean(dim=2).sum(dim=1).cpu().numpy() ####
                    # emb = fused_d[:, 0, :].sum(dim=0).cpu().numpy()  # → (D,) Use only first concept
                    all_dance_embeddings.append(emb)
                    dance_labels.extend([tag] * emb.shape[0])

                # Music embeddings
                for tag, w in [
                    (f"m({i},{i},{i},{i})", w_ii),
                    (f"m({i},{i},{j},{i})", w_ji),
                    (f"m({i},{i},{j},{j})", w_jj),
                ]:
                    _, fused_m = self._apply_concept_weights(d_i, m_i, w)
                    emb = fused_m.max(dim=2).values.sum(dim=1).cpu().numpy() ####
                    # emb = fused_m.mean(dim=2).sum(dim=1).cpu().numpy() ####
                    # emb = fused_m[:, 0, :].sum(dim=0).cpu().numpy()  # → (D,) Use only first concept
                    all_music_embeddings.append(emb)
                    music_labels.extend([tag] * emb.shape[0])


        def plot_embeddings(embeddings_list, labels, title, filename_prefix):
            embeddings_np = np.concatenate(embeddings_list, axis=0)
            labels_np = np.array(labels)

            output_dir = "./probe_results"
            os.makedirs(output_dir, exist_ok=True)

            def plot_scatter(embedding_2d, method, filename_suffix, prefix):
                plt.figure(figsize=(10, 8))
                unique_labels = sorted(set(labels_np))
                for label in unique_labels:
                    idx = labels_np == label
                    plt.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1], label=label, s=40, alpha=0.75)
                plt.title(f"{title} ({method})")
                plt.xlabel(f"{method} 1")
                plt.ylabel(f"{method} 2")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                save_path = os.path.join(output_dir, f"{prefix}_{filename_suffix}.png")
                plt.savefig(save_path, dpi=300)
                plt.close()
                print(f"[INFO] Saved {method} plot to: {save_path}")

            # PCA
            pca = PCA(n_components=2)
            pca_2d = pca.fit_transform(embeddings_np)
            plot_scatter(pca_2d, "PCA", "pca", filename_prefix)

            # t-SNE (sample if too large)
            if embeddings_np.shape[0] > 1000:
                idx = np.random.choice(embeddings_np.shape[0], 1000, replace=False)
                tsne_input = embeddings_np[idx]
                tsne_labels = labels_np[idx]
            else:
                tsne_input = embeddings_np
                tsne_labels = labels_np

            tsne = TSNE(n_components=2, perplexity=30, n_iter=500, init='random', learning_rate='auto')
            tsne_2d = tsne.fit_transform(tsne_input)
            plot_scatter(tsne_2d, "t-SNE", "tsne", filename_prefix)

        plot_embeddings(all_dance_embeddings, dance_labels, "Dance Embedding Variants", "dance_probe")
        plot_embeddings(all_music_embeddings, music_labels, "Music Embedding Variants", "music_probe")


    def rerank(self):
        self.model.eval()
        fused_music_arr, fused_video_arr = [], []
        BD_music_embed_arr, BD_vid_embed_arr = [], []

        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                for key in ['music', 'pose', 'music_beat', 'video_beat']:
                    data[key] = data[key].to(self.device)

                music_embed, video_embed = self.model(data)

                _, fused_music, fused_video = self._compute_losses(
                    music_embed['music_fuse'], video_embed['video_fuse'],
                    music_embed['music_beat'], video_embed['video_beat']
                )

                fused_music_arr.append(fused_music)    # [B, L, K, D]
                fused_video_arr.append(fused_video)   # [B, L, K, D]

                BD_music_embed_arr.append(
                    music_embed['music_fuse'].reshape(-1, int(self.config.embed_dim) * self.config.num_frames)
                )
                BD_vid_embed_arr.append(
                    video_embed['video_fuse'].reshape(-1, int(self.config.embed_dim) * self.config.num_frames)
                )

        # Combine all batches
        fused_music = torch.cat(fused_music_arr, dim=0)
        fused_video = torch.cat(fused_video_arr, dim=0)
        BD_music_embeds = torch.cat(BD_music_embed_arr, dim=0)
        BD_vid_embeds = torch.cat(BD_vid_embed_arr, dim=0)

        # Compute similarities
        refined_sims = sim_matrix_inference_per_frame_aspect_max(fused_music.unsqueeze(1), fused_video).cpu().detach()  # [V_text, 1, V_vid]
        base_sims = sim_matrix_inference(BD_music_embeds.unsqueeze(1), BD_vid_embeds).cpu().detach()                    # [V_text, 1, V_vid]

        # Reranking
        V_text = BD_music_embeds.shape[0]
        gt_indices = list(range(V_text))
        top_k_list = [k for k in [50, 100, 200, 500, 1000] if k <= V_text]    
        eval_ranks = [1, 5, 10, 50, 100]
        refined_scores = refined_sims.squeeze(1)  # [V_text, V_vid]
        base_sims = base_sims.squeeze(1)          # [V_text, V_vid]

        # Baseline for comparison
        baseline_res = self.metrics(base_sims.unsqueeze(1))
        baseline_r = {r: baseline_res[f'R{r}'] for r in eval_ranks if f'R{r}' in baseline_res}

        # Collect for plot
        rerank_r_dict = {r: [] for r in eval_ranks}
        res = {}

        for k in top_k_list:
            reranked_indices = []
            for i in range(V_text):
                topk = torch.topk(base_sims[i], k=k).indices
                topk_scores = refined_scores[i][topk]
                reranked = topk[torch.argsort(topk_scores, descending=True)]
                reranked_indices.append(reranked)
            reranked_indices = torch.stack(reranked_indices)

            for r in eval_ranks:
                if r > k:
                    rerank_r_dict[r].append(None)
                    continue
                correct = sum(gt_indices[i] in reranked_indices[i][:r] for i in range(V_text))
                recall = 100 * correct / V_text
                rerank_r_dict[r].append(recall)
                res[f'Top-{k}_R{r}'] = recall
                print(f"Top-{k}_R{r}: {recall:.2f}")

        # Plot Recall@R curves
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.flatten()
        os.makedirs(self.output_dir, exist_ok=True)

        for idx, r in enumerate(eval_ranks):
            if idx >= len(axes):
                break
            ax = axes[idx]
            if all(v is None for v in rerank_r_dict[r]):
                ax.axis('off')
                continue

            valid_k = [k for i, k in enumerate(top_k_list) if rerank_r_dict[r][i] is not None]
            valid_vals = [v for v in rerank_r_dict[r] if v is not None]

            # Plot Re-ranked points
            ax.plot(valid_k, valid_vals, 'o-', color='C0', label='Re-ranked')

            # Annotate points (shift slightly)
            for k_val, val in zip(valid_k, valid_vals):
                ax.text(k_val, val * 1.01, f"{val:.1f}", fontsize=8, ha='center', color='C0')

            # Baseline line + label
            if r in baseline_r:
                baseline_val = baseline_r[r]
                ax.axhline(baseline_val, linestyle='--', color='gray', label='Baseline')
                ax.text(valid_k[-1] + 5, baseline_val, f'{baseline_val:.1f}%', va='center', ha='left', fontsize=8, color='gray')

            ax.set_title(f"R@{r}", fontsize=11)
            ax.set_xlabel("Top-K from baseline", fontsize=9)
            ax.set_ylabel("Recall (%)", fontsize=9)
            ax.set_xticks(valid_k)
            ax.set_xticklabels([str(k) for k in valid_k])
            ax.grid(True)
            ax.legend(loc='upper left', fontsize=8, frameon=False)

            # Hide unused subplots
        for i in range(len(eval_ranks), len(axes)):
            axes[i].axis('off')

        fig.suptitle("Re-ranking vs Baseline Recall@R", fontsize=15)
        fig.subplots_adjust(top=0.9, hspace=0.5, wspace=0.3)  # Adjust spacing
        combined_plot_path = os.path.join(self.output_dir, "rerank_vs_baseline_combined.png")
        plt.savefig(combined_plot_path)
        plt.close()

        return res
