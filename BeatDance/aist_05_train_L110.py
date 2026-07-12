import os
import random
from pathlib import Path

import numpy as np
import torch
from config.all_config import CusConfig
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.loss import LossFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.optimization import AdamW, get_cosine_schedule_with_warmup
from torch.utils.tensorboard.writer import SummaryWriter
from trainer.trainer_beatdance_frame import Trainer

batch_size = 32
seglen = 128
num_frames = 128
exp_name = "MotionBERT_pos"
project_root = Path(__file__).resolve().parents[1]
config_path = project_root / "BeatDance/config/aist_seg10_L10.json"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("BeatDance contrastive pretraining requires CUDA")
    print("Current device:", torch.cuda.get_device_name(0))
    torch.multiprocessing.set_start_method("spawn", force=True)
    device = "cuda"

    config = CusConfig(str(config_path), exp_name)
    config.exp_name = exp_name
    config.batch_size = batch_size
    config.num_frames = num_frames

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

    train_data_loader = DataFactory.get_data_loader(config, split_type="train")
    valid_data_loader = DataFactory.get_data_loader(config, split_type="val")
    model = ModelFactory.get_model(config)
    model.to(device)

    metrics = {"t2v": t2v_metrics, "v2t": v2t_metrics}.get(config.metric)
    if metrics is None:
        raise ValueError(f"Unsupported metric: {config.metric}")

    params_optimizer = list(model.named_parameters())
    clip_params = [p for n, p in params_optimizer if "clip." in n]
    noclip_params = [p for n, p in params_optimizer if "clip." not in n]

    optimizer_grouped_params = [
        {"params": clip_params, "lr": config.clip_lr},
        {"params": noclip_params, "lr": config.noclip_lr},
    ]
    optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)
    num_training_steps = len(train_data_loader) * config.num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    loss = LossFactory.get_loss(config)

    trainer = Trainer(
        model,
        loss,
        metrics,
        optimizer,
        config=config,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=scheduler,
        writer=writer,
        tokenizer=None,
    )

    if config.load_epoch is not None:
        if config.load_epoch > 0:
            trainer.load_checkpoint(f"checkpoint-epoch{config.load_epoch}.pth")
            print(f"Loaded checkpoint from epoch {config.load_epoch}")
        else:
            trainer.load_checkpoint("model_best.pth")

    trainer.train(seglen)


if __name__ == "__main__":
    main()
