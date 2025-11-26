import os
import torch, math

import pandas as pd
import ujson as json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset, DataLoader
from config.base_config import Config
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import pandas as pd
import pytorch_lightning as pl
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np

HEIGHT = 1080
WIDTH = 1920

COCO_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),   # face
    (0, 5),  (5, 7), (7, 9), (5, 11),     # left upper body
    (0, 6), (6, 8), (8, 10), (6, 12),     # right upper body
    (11, 13),  (13, 15),                  # Left lowerbody
    (12, 14), (14, 16),                   # Right lowerbody
]

COCO_PARENT = [
    -1,  # 0: Nose (root, no parent)
    0,   # 1: Left Eye ← Nose
    0,   # 2: Right Eye ← Nose
    1,   # 3: Left Ear ← Left Eye
    2,   # 4: Right Ear ← Right Eye
    0,   # 5: Left Shoulder ← Nose (or sometimes Left Ear)
    0,   # 6: Right Shoulder ← Nose (or sometimes Right Ear)
    5,   # 7: Left Elbow ← Left Shoulder
    6,   # 8: Right Elbow ← Right Shoulder
    7,   # 9: Left Wrist ← Left Elbow
    8,   # 10: Right Wrist ← Right Elbow
    5,   # 11: Left Hip ← Left Shoulder (or spine if defined)
    6,   # 12: Right Hip ← Right Shoulder
    11,  # 13: Left Knee ← Left Hip
    12,  # 14: Right Knee ← Right Hip
    13,  # 15: Left Ankle ← Left Knee
    14   # 16: Right Ankle ← Right Knee
]


def angle_difference(a1, a2):
    """Compute wrapped angle difference within [-pi, pi]"""
    delta = a1 - a2
    delta = (delta + math.pi) % (2 * math.pi) - math.pi
    return delta

def compute_bone_angles(positions, bones):
    """
    Compute 2D joint orientation angles for each bone. (in radian on the image)
    positions: (T, J, 2)
    Returns: (T, num_bones)
    """
    T, J, _ = positions.shape
    angles = []
    for p1_idx, p2_idx in bones:
        vec = positions[:, p2_idx, :] - positions[:, p1_idx, :]  # (T, 2)
        bone_angle = torch.atan2(vec[:, 1], vec[:, 0])     # (T,)
        angles.append(bone_angle)
    return torch.stack(angles, dim=1)  # (T, num_bones)


class pdlDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, 
                 ids, 
                 config, 
                 full_df, 
                 sample_rate,
                 frame_rate,
                 split, 
                 lookup_df,
                 img_transforms=None, 
                 ):

        self.data_dir = config.videos_dir # /data/han_data/dance_data/beatdance
        self.img_transforms = img_transforms
        self.split_type = split
        self.frames = config.num_frames # number of temporal frames, L
        self.video_features_dir = os.path.join(self.data_dir, 'pdl_clip')
        #self.music_features_dir = os.path.join(self.data_dir, 'pdl_mert_s2_L43')
        self.music_features_dir = os.path.join(self.data_dir, 'pdl_mert_L10')
        self.video_beats_dir = os.path.join(self.data_dir, 'pdl_vitpose_beats') # per video file
        self.music_beats_dir = os.path.join(self.data_dir, 'pdl_music_beats')  #per music file
        self.pose_dir = os.path.join(self.data_dir, "pdl_pose")
        #self.beat_dim = config.num_frames #beat feature dimension
        self.full_df = full_df
        self.lookup_df = lookup_df
        self.ids = ids
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        #self.sao_rate = 21.5
        seg_len = 10
        self.embed_framerate = seg_len // self.frames # seglen/L

    def __getitem__(self, index):
        id = self.ids[index]
        row = self.full_df[self.full_df["sample_id"]==id]
        t_start = row["start"].values[0]
        t_end = row["end"].values[0]
        file_id = row["file_id"].values[0]

        if type(self.lookup_df) != type(None): # if the eval_len and model_len does not correspond, hard code the 10 to 2 conversion here
            n_seg = 5
            # find the sorted smaller segment sample_ids corresponding to each sample_id
            sf = self.lookup_df[self.lookup_df["file_id"]==file_id]
            sf_f = sf[sf["start"].values>=float(t_start)]
            sf_ff = sf_f[sf_f["end"].values<=float(t_end)]
            correspond_ids = sf_ff["sample_id"].values
            assert len(correspond_ids) == n_seg
            music_avg_feature = []
            for c_i in correspond_ids:
                mf_dir = os.path.join(self.music_features_dir, "{}.pt".format(c_i))
                mf_feat = torch.load(mf_dir)
                music_avg_feature.append(mf_feat)
            music_avg_feature = torch.stack(music_avg_feature) #(n_seg, L, 768)
            
        else:
            music_feature_dir = os.path.join(self.music_features_dir, "{}.pt".format(id))
            music_avg_feature = torch.load(music_feature_dir)



        pose_path = os.path.join(self.pose_dir, "{}.pt".format(file_id))
        pose_feature = torch.load(pose_path) # postprocessed motion features (T, 138)
        pose_feature = pose_feature[int(self.frame_rate*t_start):int(self.frame_rate*t_end), :] #extract features (T, 138)

        data = {
                #'video': video_avg_feature, #(B,L,512)
                'music': music_avg_feature, #(B,L,768)
                'pose': pose_feature, # (B,T,138)
                'id':id,
            }
        video_beat_dir = os.path.join(self.video_beats_dir, "{}.pt".format(file_id))
        music_beat_dir = os.path.join(self.music_beats_dir, "{}.pt".format(file_id))
        video_beat = torch.load(video_beat_dir)
        music_beat = torch.load(music_beat_dir)
        
        # resample sampling rate of beats to 21.5 hz
        video_beat_avg_feature = self.beat_preprocess(video_beat, self.frame_rate, self.embed_framerate, t_start, t_end) # (L, 1)
        music_beat_avg_feature = self.beat_preprocess(music_beat, self.sample_rate//512, self.embed_framerate, t_start, t_end)  #(L, 4)

        data['video_beat'] = video_beat_avg_feature #(B,L,D)
        data['music_beat'] = music_beat_avg_feature #(B,L,D)
        return data
    
    def __len__(self):
        return len(self.ids)


    def aggregate_clip(self, frame_features, L):
        num_frames, dC = frame_features.shape
        interval_size = max(1, num_frames // L)
        # Divide into L intervals and compute average per interval
        dance_feature = torch.stack([
            frame_features[i * interval_size:(i + 1) * interval_size].mean(dim=0) 
            for i in range(L)
        ])
        return dance_feature


    # to make video sequences evenly distributed in 30 portions
    def clip_preprocess(self, feature, t_start, t_end):
        # CLIP feature shape: (T, 1, 512)
        feature = feature.squeeze()[int(t_start*self.frame_rate):int(t_end*self.frame_rate), :] # select time
        avg_feature = self.aggregate_clip(feature, self.frames)
        return avg_feature

    # Process music beat features into fixed-size vectors, from kimura-san's code
    #principle: keep as much resolution as possible, reshape the vector to (L, T//L)
    def beat_preprocess(self, beat_indices, fps, target_fps, start, end):
        # select relevant beat from the precomputed file
        beat_indices = beat_indices[beat_indices >= int(start*fps)]
        beat_indices = beat_indices[beat_indices < int(end*fps)]
        # zero calibrate the indices
        beat_indices = beat_indices - int(start*fps)
        # compute maximum beat dim allowed for current fps
        beat_dim = fps // target_fps # 1 for dance and 4 for music
        if beat_dim == 1:
            beat_dim = 2 #upsample if beat_dim is only 1
        # convert to beat presence vector
        beats_vector = torch.zeros(int(self.frames*beat_dim))
        # resample the beat_indices (number of new samples / number of og samples)
        scale = self.frames * beat_dim / ((end-start) * fps)
        beat_indices = torch.round(beat_indices * scale).long()
        beat_indices[beat_indices>=len(beats_vector)] = len(beats_vector) - 1 # safty precaution to avoid exceeding length
        try:
            beats_vector[beat_indices] = 1 #(along beat_dim, every vector is the same)
        except:
            import pdb
            pdb.set_trace()
        return beats_vector.reshape((self.frames, int(beat_dim))) #reshape to (L, beat_dim)

    def vitpose_preprocess(self, pose_feature, t_start, t_end):
        """
        Input: 2d keypoints in COCO format (T, 17, 2)
        Output: [root position, 
                root velocity, 
                joint position (rel to root),
                joint velocity (rel to root),
                joint acceleration (rel to root),
                joint angles (each bone)
                joint angular velocity (each bone)
          ]
        """

        T, J, _ = pose_feature.shape # (T,J,2)
        # using coco format, extract joint position, velocity, acceleration in root space
        root = (pose_feature[:, 11, :] + pose_feature[:, 12, :]) / 2 # (T, 2) midpoint between left and right hip
        root_vel = torch.cat((torch.zeros(1, 2), root[1:] - root[:-1]), 0) #(T, 2)
        joint_pos = pose_feature - root.unsqueeze(1) #(T, J, 2) joint position in root space
        joint_vel = torch.cat((torch.zeros(1, J, 2), joint_pos[1:] - joint_pos[:-1]), 0) #(T, J, 2) joint linear velocity in root space
        joint_acc = torch.cat((torch.zeros(1, J, 2), joint_vel[1:] - joint_vel[:-1]), 0) #(T, J, 2) joint linear acceleration in root space
        # 2d joint orientations, from each parent joint
        # Bone angles on the 2D image
        joint_angles = compute_bone_angles(joint_pos, COCO_BONES)  # (T, num_bones)
        # Bone angular velocity (wrapped between -pi and pi) and pad zeros
        joint_angle_vel = torch.cat((torch.zeros(1, len(COCO_BONES)), angle_difference(joint_angles[1:], joint_angles[:-1])), 0)  # (T, num_bones)
        all_feat = torch.cat((root, # (T, 4+6J+2(J-1)) = (T, 138)
                              root_vel,  
                              joint_pos.flatten(start_dim=1), 
                              joint_vel.flatten(start_dim=1), 
                              joint_acc.flatten(start_dim=1), 
                              joint_angles, 
                              joint_angle_vel), 1)
        
        return all_feat[int(t_start*self.frame_rate):int(t_end*self.frame_rate), :]  # shape: (T, 138)

def curate_testset(pt_dir, df, num):
    # input: filtered df of the given split
    # pick random "num" sample_ids, such that they have unique file ids
    file_candidates = np.unique(np.array(df['file_id'].values))
    chosen_files = np.random.choice(file_candidates, num)
    #choose middle segment
    chosen_samples = []
    for c_f in chosen_files:
        samples = df[df["file_id"]==int(c_f)]["sample_id"].values
        for s in samples[len(samples)//2:]: # start from middle and go through to the end
            if s not in chosen_samples:
                chosen_samp = s
                break
        if os.path.exists(os.path.join(pt_dir, "{}.pt".format(chosen_samp))):
            chosen_samples.append(chosen_samp)
    return chosen_samples

class PdLDataModule(pl.LightningDataModule):
    def __init__(self,
                 config, 
                 full_df, # df corresponding to the loaded segment len
                 batch_size,
                 sample_rate=48000, # should be 44100 but i fucked up, so everything is a bit shifted 
                 frame_rate = 25,
                 lookup_df=None, # df of the model segment len (2s) 
                 num_workers=0):
        super().__init__()
        self.full_df = full_df
        self.lookup_df = lookup_df
        self.config = config
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.num_workers = num_workers
        self.setup()

    def setup(self, stage=None):

        train_ids = self.full_df[self.full_df["split"]=="train"]['sample_id'].values #sorted by id
        test_ids = self.full_df[self.full_df["split"]=="test"]['sample_id'].values
        val_ids = self.full_df[self.full_df["split"]=="val"]['sample_id'].values
        
        print("currently available sounds:", len(train_ids), len(test_ids), len(val_ids))
        self.trainset = pdlDataset(train_ids,
                                    self.config,
                                    self.full_df,
                                    self.sample_rate, 
                                    self.frame_rate,
                                    split="train", 
                                    lookup_df=self.lookup_df
                                )

        self.valset = pdlDataset(val_ids,
                                    self.config,
                                    self.full_df,
                                    self.sample_rate, 
                                    self.frame_rate,
                                    split="val", 
                                    lookup_df=self.lookup_df
                                )

        self.testset = pdlDataset(test_ids,
                                    self.config,
                                    self.full_df,
                                    self.sample_rate, 
                                    self.frame_rate,
                                    split="test", 
                                    lookup_df=self.lookup_df
                                )


    def collate_batch(self, batch):
        """
        metadata needs to output a list of dictionaries with key value being (conditioning type: conditioner)
        """
        
        music = torch.stack([s['music'] for s in batch]) #(B,L,768 or B,n_seg, L, 768)
        #video = torch.stack([s['video'] for s in batch])
        pose = torch.stack([s['pose'] for s in batch])
        music_beat = torch.stack([s['music_beat'] for s in batch])
        video_beat = torch.stack([s['video_beat'] for s in batch])
        ids = torch.tensor([s['id'] for s in batch])
        
        return {'music': music, 
                #'video': video,
                'pose': pose,
                'music_beat': music_beat,
                'video_beat': video_beat,
                'id': ids,
                }

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=self.collate_batch,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          collate_fn=self.collate_batch,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          collate_fn=self.collate_batch,
                          num_workers=self.num_workers)
    
   