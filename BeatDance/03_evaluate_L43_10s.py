import os
import torch
import random
import numpy as np
from config.all_config import CusConfig
from torch.utils.tensorboard.writer import SummaryWriter
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.loss import LossFactory
from trainer.trainer_beatdance_frame import Trainer
from modules.optimization import AdamW, get_cosine_schedule_with_warmup
import pdb
from tqdm import tqdm
import pandas as pd

# beat dimension

seglen = 2
totallen = 10
L = 43

def main():
    print("Current device: ", torch.cuda.get_device_name(0))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # limit to first gpu
    torch.multiprocessing.set_start_method('spawn')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_test = CusConfig("config/pdl_seg10_L43.json")
    config_train = CusConfig("config/pdl_seg2_L43.json")
    config_train.num_frames = L
    config_test.num_frames = L * (totallen // seglen)

    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config_train.no_tensorboard:
        writer = SummaryWriter(log_dir=config_train.tb_log_dir)
    else:
        writer = None


    if config_train.seed >= 0:
        torch.manual_seed(config_train.seed)
        np.random.seed(config_train.seed)
        torch.cuda.manual_seed_all(config_train.seed)
        random.seed(config_train.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_data_loader  = DataFactory.get_data_loader(config_test, split_type='train',
                                                    lookup_dir=config_train.ann_dir) # stacked 2s segments into 10s in total
    test_data_loader  = DataFactory.get_data_loader(config_test, split_type='test', 
                                                    lookup_dir=config_train.ann_dir) # 10s segments instead of 2s
    val_data_loader  = DataFactory.get_data_loader(config_test, split_type='val', 
                                                    lookup_dir=config_train.ann_dir) # 10s segments instead of 2s
    model = ModelFactory.get_model(config_train)
    model.to(device)
    
    if config_train.metric == 't2v':
        metrics = t2v_metrics
    elif config_train.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented
    

    loss = LossFactory.get_loss(config_train)

    trainer = Trainer(model, loss, metrics, None,
                      config=config_train,
                      train_data_loader=None,
                      valid_data_loader=test_data_loader,
                      lr_scheduler=None,
                      writer=writer,
                      qb_norm='QB-Norm',
                      tokenizer=None)
    
    trainer.load_checkpoint("model_best.pth")   

    # evaluate the model on test set 
    #trainer.validate_qbnorm(model_len=seglen, eval_len=totallen)
    
    trainer.validate(model_len=seglen, eval_len=totallen)
   


if __name__ == '__main__':
    main()
