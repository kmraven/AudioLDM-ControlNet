from config.base_config import Config
# from datasets.model_transforms import init_transform_dict
# from datasets.msrvtt_dataset import MSRVTTDataset
# from datasets.msvd_dataset import MSVDDataset
# from datasets.lsmdc_dataset import LSMDCDataset
from datasets.pdl_dataset import PdLDataModule
from datasets.musiconly_dataset import MusicOnlyModule
from datasets.aistpp_dataset import aistppDataModule
import pandas as pd
# from torch.utils.data import DataLoader

class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type='train', lookup_dir=None):
        # img_transforms = init_transform_dict(config.input_res)
        # train_img_tfms = img_transforms['clip_train']
        # test_img_tfms = img_transforms['clip_test']

        # if config.dataset_name == "MSRVTT":
        #     if split_type == 'train':
        #         dataset = MSRVTTDataset(config, split_type, train_img_tfms)
        #         return DataLoader(dataset, batch_size=config.batch_size,
        #                    shuffle=True, num_workers=config.num_workers)
        #     else:
        #         dataset = MSRVTTDataset(config, split_type, test_img_tfms)
        #         return DataLoader(dataset, batch_size=config.batch_size,
        #                    shuffle=False, num_workers=config.num_workers)

        # elif config.dataset_name == "MSVD":
        #     if split_type == 'train':
        #         dataset = MSVDDataset(config, split_type, train_img_tfms)
        #         return DataLoader(dataset, batch_size=config.batch_size,
        #                 shuffle=True, num_workers=config.num_workers)
        #     else:
        #         dataset = MSVDDataset(config, split_type, test_img_tfms)
        #         return DataLoader(dataset, batch_size=config.batch_size,
        #                 shuffle=False, num_workers=config.num_workers)
        # elif config.dataset_name == 'LSMDC':
        #     if split_type == 'train':
        #         dataset = LSMDCDataset(config, split_type, train_img_tfms)
        #         return DataLoader(dataset, batch_size=config.batch_size,
        #                     shuffle=True, num_workers=config.num_workers)
        #     else:
        #         dataset = LSMDCDataset(config, split_type, test_img_tfms)
        #         return DataLoader(dataset, batch_size=config.batch_size,
        #                     shuffle=False, num_workers=config.num_workers)
        # elif config.dataset_name == 'pdl':
        if config.dataset_name == 'pdl':
            df = pd.read_csv(config.ann_dir)
            if lookup_dir:
                lookup_df = pd.read_csv(lookup_dir)
            else:
                lookup_df = None
            pdldata = PdLDataModule(config, df,
                                    batch_size=config.batch_size,
                                    lookup_df=lookup_df)
            if split_type == 'train':
                return pdldata.train_dataloader()
            elif split_type == 'test':
                return pdldata.test_dataloader()
            else:
                return pdldata.val_dataloader()
        elif config.dataset_name == "musiconly":
            df = pd.read_csv(config.ann_dir)
            modata = MusicOnlyModule(config, df,
                                     batch_size=config.batch_size)
            if split_type == 'train':
                return modata.train_dataloader()
            elif split_type == 'test':
                return modata.test_dataloader()
            else:
                return modata.val_dataloader()
        elif config.dataset_name == "aist":
            aistdata = aistppDataModule(
                data_dir=config.videos_dir,
                batch_size=config.batch_size,
            )
            if split_type == 'train':
                return aistdata.train_dataloader()
            elif split_type == 'test':
                return aistdata.test_dataloader()
            else:
                return aistdata.val_dataloader()
        else:
            raise NotImplementedError
