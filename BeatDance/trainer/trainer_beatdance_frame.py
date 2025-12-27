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
    sim_matrix_training_per_frame,
    sim_matrix_inference,
    sim_matrix_inference_per_frame,
    generate_embeds_per_video_id,
    beat_similarity,
    qb_norm
)

class Trainer(BaseTrainer):
    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader,
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None,
                 qb_norm=None):
        super().__init__(model, loss, metrics, optimizer, config, writer)

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.tokenizer = tokenizer
        self.lr_scheduler = lr_scheduler

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -400.0 # changed to much smaller since my val set is 2k+ samples, difficult to achieve -1 score.
        self.best = -400.0

        self.qb_norm = qb_norm
        self.qbnorm_beta = config.qbnorm_beta
        self.qbnorm_k = config.qbnorm_k
        self.metric = config.metric
        self.L = config.num_frames


    def _compute_concept_embedding(self, music_embed, video_embed):
        """
        Args:
            music_embed: (B, L, D)
            video_embed: (B, L, D)
        Returns:
            sim_loss, embed_norm, mask_norm, embedded_music, embedded_video
            all with shape or scalar consistent to [B, L, D]
        """
        B, L, D = music_embed.shape

        music_feat = music_embed / (torch.norm(music_embed, p=2, dim=-1, keepdim=True) + 1e-10)
        video_feat = video_embed / (torch.norm(video_embed, p=2, dim=-1, keepdim=True) + 1e-10)

        concept_input = torch.cat([music_feat, video_feat], dim=-1).view(B * L, -1)
        concept_weights = self.concept_branch(concept_input).view(B, L, self.num_concepts)

        music_flat = music_embed.view(B * L, D)
        video_flat = video_embed.view(B * L, D)

        embedded_music, embedded_video = None, None
        mask_norm = None
        # sim_loss = 0.0

        for i in range(self.num_concepts):
            concept_idx = torch.full((B * L,), i, dtype=torch.long, device=self.device)

            m_emb, m_mask, m_norm, _ = self.conditional_sim(music_flat, concept_idx)
            v_emb, v_mask, v_norm, _ = self.conditional_sim(video_flat, concept_idx)

            if mask_norm is None:
                mask_norm = v_mask
            else:
                mask_norm = mask_norm + v_mask

            weight = concept_weights[:, :, i].reshape(B * L, 1)
            m_weighted = m_emb * weight
            v_weighted = v_emb * weight


            if embedded_music is None:
                embedded_music = m_weighted
                embedded_video = v_weighted 
            else:
                embedded_music = embedded_music + m_weighted
                embedded_video = embedded_video + v_weighted

        embedded_music = embedded_music.view(B, L, D)
        embedded_video = embedded_video.view(B, L, D)

        sim = sim_matrix_training_per_frame(embedded_music, embedded_video)
        sim_loss = self.loss(sim, self.model.clip_logit_scale)

        mask_norm /= self.num_concepts
        embed_norm = (m_norm + v_norm) / 2  

        return sim_loss, embed_norm, mask_norm, embedded_music, embedded_video


    def _train_epoch(self, epoch, seglen):
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps - 1, self.evals_per_epoch + 1, dtype=int)[1:]
        #print("eval steps", eval_steps, num_steps)
        #print("training epoch started", time.time()-start)
        for batch_idx, data in enumerate(self.train_data_loader):
            # music: [32, L, 768], video: [32, L, 512], pose: [32, T, 138]
            # music_beat: [32, L, L], video_beat: [32, L, L]
            for key in ['music', 'video', 'music_beat', 'video_beat']:
                data[key] = data[key].to(self.device)

            # all features in music embed: [32, L, 256] video embed: [32, L, 256]
            music_embed, video_embed = self.model(data, 'train')

            # BeatDance Only
            output = sim_matrix_training_per_frame(music_embed['music_fuse'], video_embed['video_fuse']) #(B,B)
            output1 = sim_matrix_training_per_frame(music_embed['music_beat'], video_embed['video_beat'])
            # weighted sum of negative likelihood of mean of frame-wise similarity measure?
            loss = self.loss(output, self.model.clip_logit_scale) + self.beta * self.loss(output1, self.model.clip_logit_scale) # with beatloss

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
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps - 1, seglen, seglen)
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
            for batch_idx, data in tqdm(enumerate(self.valid_data_loader)): #EVERY DATA IS A BATCH OF FEATURES
                for key in ['music', 'video', 'music_beat', 'video_beat']:
                    data[key] = data[key].to(self.device)

                if model_len != eval_len: # concatenate the features together
                    music_embed, video_embed = self.concat_output(model_len, eval_len, data)
                else:
                    music_embed, video_embed = self.model(data) # music_embed['music_fuse'] # EACH FEATURE WITHIN IS (B, L, D)

                #BeatDance Only
                output = sim_matrix_training_per_frame(music_embed['music_fuse'], video_embed['video_fuse']) #(B, B) NORMALIZED DOT PRODUCT, COSINE SIMILARITY
                output1 = sim_matrix_training_per_frame(music_embed['music_beat'], video_embed['video_beat']) #(B, B)
                loss = self.loss(output, self.model.clip_logit_scale) + self.beta * self.loss(output1, self.model.clip_logit_scale) # TAKE THE MEAN DIAGONAL OF SOFTMAXED COSINE SIMILARITY
                total_val_loss += loss.item()
                fused_music_arr.append(music_embed['music_fuse'])   
                fused_video_arr.append(video_embed['video_fuse'])  

        # Combine all batches: [V, L, D], WHERE V IS ALL SAMPLES IN THE VALIDATION SET
        fused_music = torch.cat(fused_music_arr, dim=0)
        fused_video = torch.cat(fused_video_arr, dim=0)

        sims = sim_matrix_inference_per_frame(fused_music.unsqueeze(1), fused_video).cpu().detach() # sims shape: [V_text, T, V_vid, L]
      
        total_val_loss /= len(self.valid_data_loader) # for each pair, take the mean
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
        import pdb
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                for key in ['music', 'video', 'music_beat', 'video_beat']:
                    data[key] = data[key].to(self.device)
                if model_len != eval_len: # concatenate the features together
                    music_embed, video_embed = self.concat_output(model_len, eval_len, data)
                else:
                    music_embed, video_embed = self.model(data) # music_embed['music_fuse']

                #BeatDance Only
                output = sim_matrix_training_per_frame(music_embed['music_fuse'], video_embed['video_fuse'])
                output1 = sim_matrix_training_per_frame(music_embed['music_beat'], video_embed['video_beat'])
                loss = self.loss(output, self.model.clip_logit_scale) + self.beta * self.loss(output1, self.model.clip_logit_scale)
                total_val_loss += loss.item()
                fused_music_arr.append(music_embed['music_fuse'])   
                fused_video_arr.append(video_embed['video_fuse'])  

        # Combine all batches: [V, L, D]
        fused_music = torch.cat(fused_music_arr, dim=0)
        fused_video = torch.cat(fused_video_arr, dim=0)

        def _get_train_embeds(direction, model_len, eval_len):
            embed_list = []
            for _, data in tqdm(enumerate(self.train_data_loader)):
                for key in ['music', 'video', 'music_beat', 'video_beat']:
                    data[key] = data[key].to(self.device)
                if model_len != eval_len: # concatenate the features together
                    music_out, video_out = self.concat_output(model_len, eval_len, data)
                else:
                    music_out, video_out = self.model(data)
                if direction == 'v2t':
                    embed_list.append(video_out['video_fuse'])  # [B, F, D]
                else:
                    embed_list.append(music_out['music_fuse'])  # [B, F, D]
            return torch.cat(embed_list, dim=0)  # [V_train, F, D]

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
                test_embed_sims = sim_matrix_inference_per_frame(fused_video.unsqueeze(1), fused_music).cpu().detach()
                test_embed_sims = test_embed_sims.max(dim=-1).values
                train_embed = _get_train_embeds('v2t', model_len, eval_len)
                train_embed_sims = sim_matrix_inference_per_frame(train_embed.unsqueeze(1), fused_music).cpu().detach()
                train_embed_sims = train_embed_sims.max(dim=-1).values
                sims = _apply_qbnorm(train_embed_sims, test_embed_sims, transpose=True)
            else:
                test_embed_sims = sim_matrix_inference_per_frame(fused_music.unsqueeze(1), fused_video).cpu().detach()
                test_embed_sims = test_embed_sims.max(dim=-1).values
                train_embed = _get_train_embeds('t2v', model_len, eval_len)
                train_embed_sims = sim_matrix_inference_per_frame(train_embed.unsqueeze(1), fused_video).cpu().detach()
                train_embed_sims = train_embed_sims.max(dim=-1).values
                sims = _apply_qbnorm(train_embed_sims, test_embed_sims, transpose=False)

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

    def concat_output(self, model_len, eval_len, data):
        n_seg = eval_len // model_len
        music_fuse = []
        music_beat = []
        video_fuse = []
        video_beat = []
        for i in range(n_seg):
            data_seg = {}
            data_seg["music"] = data["music"][:,i,:,:]
            data_seg["video"] = data["video"][:,(i*model_len*25):((i+1)*model_len*25),:]
            data_seg["music_beat"] = data["music_beat"][:,(i*self.L):((i+1)*self.L),:]
            data_seg["video_beat"] = data["video_beat"][:,(i*self.L):((i+1)*self.L),:]
            m_eb_seg, v_eb_seg = self.model(data_seg)
            music_fuse.append(m_eb_seg["music_fuse"])#(B,L,256)
            music_beat.append(m_eb_seg["music_beat"])#(B,L,256)
            video_fuse.append(v_eb_seg["video_fuse"])#(B,L,256)
            video_beat.append(v_eb_seg["video_beat"])#(B,L,256)
        music_embed = {}
        video_embed = {}
        music_embed["music_fuse"] = torch.concat(music_fuse, dim=1)
        music_embed["music_beat"] = torch.concat(music_beat, dim=1)
        video_embed["video_fuse"] = torch.concat(video_fuse, dim=1)
        video_embed["video_beat"] = torch.concat(video_beat, dim=1)
        return music_embed, video_embed