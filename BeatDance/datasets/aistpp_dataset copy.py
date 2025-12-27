import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class aistppDataset(Dataset):
    """
    data_dir: directory where all videos are stored
    split_type: 'train'/'val'/'test'
    """
    def __init__(
            self,
            ids,
            data_dir,
            split,
        ):
        self.ids = ids
        self.music_features_dir = os.path.join(data_dir, 'music_feature', split)
        self.music_beats_dir = os.path.join(data_dir, 'music_beat', split)
        self.pose_dir = os.path.join(data_dir, 'video_feature', split)
        self.video_beats_dir = os.path.join(data_dir, 'video_beat', split)

    def __getitem__(self, index):
        id = self.ids[index]
        data = {
            'id': id,
            'music': torch.load(os.path.join(self.music_features_dir, f"{id}.pt")),  #(B,L,768)
            'pose': torch.load(os.path.join(self.pose_dir, f"{id}.pt")),  # (B,T,138)
            'music_beat': torch.load(os.path.join(self.music_beats_dir, f"{id}.pt")),  # (B,L,D)
            'video_beat': torch.load(os.path.join(self.video_beats_dir, f"{id}.pt")),  # (B,L,D)
        }
        return data

    def __len__(self):
        return len(self.ids)


class aistppDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir,  # /data/rkimura/aist++/beatdance_features
            batch_size,
            num_workers=0,
        ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trainset = aistppDataset(
            self.__list_filenames(os.path.join(data_dir, "music_feature", "train"), '.pt'),
            data_dir,
            split="train"
        )
        self.valset = aistppDataset(
            self.__list_filenames(os.path.join(data_dir, "music_feature", "val"), '.pt'),
            data_dir,
            split="val"
        )
        self.testset = aistppDataset(
            self.__list_filenames(os.path.join(data_dir, "music_feature", "test"), '.pt'),
            data_dir,
            split="test"
        )

    def __list_filenames(self, root_dir: str, suffix: str) -> list[str]:
        # mBR0/gBR_sBM_c01_d04_mBR0_ch02_1.pkl -> mBR0/gBR_sBM_c01_d04_mBR0_ch02_1
        result = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if fn.endswith(suffix):
                    full_path = os.path.join(dirpath, fn)
                    rel_path = os.path.relpath(full_path, root_dir)
                    result.append(rel_path.replace(suffix, ''))
        return result

    def collate_batch(self, batch):
        """
        metadata needs to output a list of dictionaries with key value being (conditioning type: conditioner)
        """
        music = torch.stack([s['music'] for s in batch]) #(B,L,768 or B,n_seg, L, 768)
        pose = torch.stack([s['pose'] for s in batch])
        music_beat = torch.stack([s['music_beat'] for s in batch])
        video_beat = torch.stack([s['video_beat'] for s in batch])
        ids = [s['id'] for s in batch]
        return {
            'music': music,
            'pose': pose,
            'music_beat': music_beat,
            'video_beat': video_beat,
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
