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

# beat dimension

data_dir = "/data/han_data/dance_data/beatdance"
output_dir = "/data/han_data/dance_data/beatdance/models"
ann_dir = "/data/han_data/dance_data/pdl_chopped_10.csv"
os.makedirs(output_dir, exist_ok=True)
batch_size = 32
seglen = 10
L = 10
exp_name = f"seglen{seglen}_L{L}_batchsize{batch_size}_framewise"

def main():
    print("Current device: ", torch.cuda.get_device_name(0))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # limit to first gpu
    torch.multiprocessing.set_start_method('spawn')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = CusConfig("config/aist_seg10_L10.json", exp_name)
    config.exp_name = exp_name
    config.videos_dir = data_dir
    config.batch_size = batch_size
    config.num_frames = L
    config.output_dir = output_dir
    config.ann_dir = ann_dir


    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None


    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config.huggingface:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    else:
        from modules.tokenization_clip import SimpleTokenizer
        tokenizer = SimpleTokenizer()

    train_data_loader = DataFactory.get_data_loader(config, split_type='train')
    valid_data_loader  = DataFactory.get_data_loader(config, split_type='val')
    model = ModelFactory.get_model(config)
    model.to(device)

    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented

    params_optimizer = list(model.named_parameters())
    clip_params = [p for n, p in params_optimizer if "clip." in n]
    noclip_params = [p for n, p in params_optimizer if "clip." not in n]

    optimizer_grouped_params = [
        {'params': clip_params, 'lr': config.clip_lr},
        {'params': noclip_params, 'lr': config.noclip_lr}
    ]
    optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)
    num_training_steps = len(train_data_loader) * config.num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    loss = LossFactory.get_loss(config)

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=scheduler,
                      writer=writer,
                      tokenizer=tokenizer)

    if config.load_epoch is not None:
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
            print("loaded checkpoint from epoch {}".format(config.load_epoch))
        else:
            trainer.load_checkpoint("model_best.pth")

    trainer.train(seglen)

if __name__ == '__main__':
    main()
