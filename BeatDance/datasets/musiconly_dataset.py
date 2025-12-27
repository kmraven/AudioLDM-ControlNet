import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import os
import torch

class MusicOnlyData(Dataset):
    def __init__(self, 
                 ids,
                 df,
                 config,
                 sample_rate=44100, 
                 seg_len=10,
                 split="train", 
                 ):
        self.data_dir = config.videos_dir
        self.ids = ids
        self.df = df
        self.sample_rate = sample_rate
        self.music_features_dir = os.path.join(self.data_dir, 'mo_mert_L10')
        self.music_beats_dir = os.path.join(self.data_dir, 'mo_music_beats')  #per music file
        self.seg_len = seg_len
        self.frames = seg_len
        self.split = split
        self.sao_rate = 21.5
        self.embed_framerate = seg_len // self.frames #self.sao_rate

        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):

        # need MERT and music beat features
        id = self.ids[idx]
        row = self.df[self.df["sample_id"]==id]
        t_start = row["start_s"].values[0]
        t_end = row["end_s"].values[0]

        music_feature_dir = os.path.join(self.music_features_dir, "{}.pt".format(id))
        music_avg_feature = torch.load(music_feature_dir)
        music_beat_dir = os.path.join(self.music_beats_dir, "{}.pt".format(id))
        music_beat = torch.load(music_beat_dir)
        music_beat_avg_feature = self.beat_preprocess(music_beat, self.sample_rate//512, self.embed_framerate, t_start, t_end)  #(L, 4)

        data = {
        'music': music_avg_feature, #(B,L,768)
        'id':id,
        'music_beat': music_beat_avg_feature
        }
        return data
    
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

    
class MusicOnlyModule(pl.LightningDataModule):
    def __init__(self,
                config,
                 full_df, # pdl_chopped_5.12.csv
                 batch_size,
                 sample_rate=44100, 
                 num_workers=0):
        super().__init__()
        self.full_df = full_df
        self.config = config
        self.batch_size = batch_size
        self.seg_len = config.num_frames
        self.sample_rate = sample_rate
        self.num_workers = num_workers
        self.setup()

    def setup(self, stage=None):

        train_ids = self.full_df[self.full_df["split"]=="train"]['sample_id'].values #sorted by id
        test_ids = self.full_df[self.full_df["split"]=="test"]['sample_id'].values
        val_ids = self.full_df[self.full_df["split"]=="val"]['sample_id'].values

        self.trainset = MusicOnlyData(train_ids,
                                    self.full_df,
                                    self.config, 
                                    self.sample_rate, 
                                    self.seg_len, 
                                    split="train", 
                                )

        self.valset = MusicOnlyData(val_ids,
                                    self.full_df,
                                    self.config, 
                                    self.sample_rate, 
                                    self.seg_len,
                                    split="val", 
                                )

        self.testset = MusicOnlyData(test_ids,
                                    self.full_df,
                                    self.config, 
                                    self.sample_rate, 
                                    self.seg_len,
                                    split="test", 
                                )


    def collate_batch(self, batch):
        """
        metadata needs to output a list of dictionaries with key value being (conditioning type: conditioner)
        """
        
        music = torch.stack([s['music'] for s in batch]) #(B,L,768 or B,n_seg, L, 768)
        music_beat = torch.stack([s['music_beat'] for s in batch])
        ids = torch.tensor([s['id'] for s in batch])
        
        return {'music': music, 
                'music_beat': music_beat,
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
                          shuffle=True,
                          drop_last=True,
                          collate_fn=self.collate_batch,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          drop_last=False,
                          collate_fn=self.collate_batch,
                          num_workers=self.num_workers)
    

# for aist++ experiment
class MusicOnlyDataset(Dataset):
    """
    data_dir: directory where all videos are stored
    split_type: 'train'/'val'/'test'
    """
    def __init__(
            self,
            data_dir,  # /data/rkimura/aist++/beatdance_features
            metadata,
            split,
        ):
        self.metadata = metadata[metadata['split']==split]
        self.split = split
        self.ids = self.metadata['sample_id'].tolist()
        self.music_features_dir = os.path.join(data_dir, 'fm_tensor')
        self.music_beats_dir = os.path.join(data_dir, 'fbm_tensor')

    def __getitem__(self, index):
        id = self.ids[index]
        data = {
            'id': id,
            'music': torch.load(os.path.join(self.music_features_dir, f"{id}.pt")),  #(B,L,768)
            'music_beat': torch.load(os.path.join(self.music_beats_dir, f"{id}.pt")),  # (B,L,D)
        }
        assert torch.isnan(data['music']).any() == False, "NaN in music features for id {}".format(id)
        assert torch.isfinite(data['music']).all(), "Inf in music features for id {}".format(id)
        assert torch.isnan(data['music_beat']).any() == False, "NaN in music beat features for id {}".format(id)
        assert torch.isfinite(data['music_beat']).all(), "Inf in music beat features for id {}".format(id)
        return data
    
    def __len__(self):
        return len(self.ids)


class MusicOnlyDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir,  # /data/rkimura/aist++/beatdance_features
            metadata,
            batch_size,
            num_workers=0,
        ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trainset = MusicOnlyDataset(
            data_dir,
            metadata,
            split="train"
        )
        self.valset = MusicOnlyDataset(
            data_dir,
            metadata,
            split="val"
        )
        self.testset = MusicOnlyDataset(
            data_dir,
            metadata,
            split="test"
        )

    def collate_batch(self, batch):
        """
        metadata needs to output a list of dictionaries with key value being (conditioning type: conditioner)
        """
        music = torch.stack([s['music'] for s in batch]) #(B,L,768 or B,n_seg, L, 768)
        music_beat = torch.stack([s['music_beat'] for s in batch])
        ids = [s['id'] for s in batch]
        return {
            'music': music,
            'music_beat': music_beat,
            'id': ids,
        }

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_batch,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_batch,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_batch,
            num_workers=self.num_workers
        )
