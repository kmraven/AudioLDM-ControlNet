import os
import sys
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "true"
sys.path.append("src")
warnings.simplefilter(action="ignore", category=FutureWarning)

import shutil
import argparse
import yaml
import torch

from tqdm import tqdm
from pytorch_lightning.strategies.ddp import DDPStrategy
from huggingface_hub import hf_hub_download
from audioldm_train.utilities.data.dataset_aistpp import AISTBeatDanceDataset

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from audioldm_train.utilities.tools import (
    get_restore_step,
    copy_test_subset_data,
)
from audioldm_train.utilities.model_util import instantiate_from_config
from audioldm_train.train.latent_diffusion_controlnet import modify_state_dict, download_checkpoint
import logging

logging.basicConfig(level=logging.WARNING)


use_text_condition = True


def main(configs, config_yaml_path, exp_group_name, exp_name, perform_validation):
    if "seed" in configs.keys():
        seed_everything(configs["seed"])
    else:
        print("SEED EVERYTHING TO 0")
        seed_everything(0)

    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(
            configs["precision"]
        )  # highest, high, medium

    log_path = configs["log_directory"]
    batch_size = configs["model"]["params"]["batchsize"]

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    val_dataset = AISTBeatDanceDataset(configs, split="test", add_ons=dataloader_add_ons, use_text_condition=use_text_condition)
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
    )
    print(
        "The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s"
        % (len(val_dataset), len(val_loader), batch_size)
    )

    # Copy test data
    test_data_subset_folder = os.path.join(
        os.path.dirname(configs["log_directory"]),
        "testset_data",
        val_dataset.dataset_name,
    )
    os.makedirs(test_data_subset_folder, exist_ok=True)
    copy_test_subset_data(val_dataset.data, test_data_subset_folder)

    try:
        config_reload_from_ckpt = configs["reload_from_ckpt"]
    except:
        config_reload_from_ckpt = None

    try:
        limit_val_batches = configs["step"]["limit_val_batches"]
    except:
        limit_val_batches = None

    validation_every_n_epochs = configs["step"]["validation_every_n_epochs"]
    save_checkpoint_every_n_steps = configs["step"]["save_checkpoint_every_n_steps"]
    max_steps = configs["step"]["max_steps"]

    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")

    wandb_path = os.path.join(log_path, exp_group_name, exp_name)

    os.makedirs(checkpoint_path, exist_ok=True)
    shutil.copy(config_yaml_path, wandb_path)

    if config_reload_from_ckpt is not None:
        resume_from_checkpoint = config_reload_from_ckpt
        print("Reload ckpt specified in the config file %s" % resume_from_checkpoint)
    else:
        print("Train from scratch")
        resume_from_checkpoint = None

    devices = torch.cuda.device_count()

    latent_diffusion = instantiate_from_config(configs["model"])
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)

    latent_diffusion.test_data_subset_path = test_data_subset_folder

    print("==> Save checkpoint every %s steps" % save_checkpoint_every_n_steps)
    print("==> Perform validation every %s epochs" % validation_every_n_epochs)

    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        max_steps=max_steps,
        num_sanity_val_steps=1,
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch=validation_every_n_epochs,
        strategy=DDPStrategy(find_unused_parameters=True),
    )

    try:
        ckpt = torch.load(resume_from_checkpoint)["state_dict"]
    except FileNotFoundError:
        ckpt_path = download_checkpoint(resume_from_checkpoint)
        ckpt = torch.load(ckpt_path)["state_dict"]

    if not any("controlnet" in key for key in ckpt.keys()):
        modify_dict = {
            "cond_stage_model": {
                "new_key": "cond_stage_models.0",
                "duplicate": False,
            },
            "model.diffusion_model": {
                "new_key": "controlnet_stage_models.0",
                "duplicate": True,
            }
        }
        ckpt = modify_state_dict(ckpt, modify_dict)

    key_not_in_model_state_dict = []
    size_mismatch_keys = []
    state_dict = latent_diffusion.state_dict()
    print("Filtering key for reloading:", resume_from_checkpoint)
    print(
        "State dict key size:",
        len(list(state_dict.keys())),
        len(list(ckpt.keys())),
    )
    for key in tqdm(list(ckpt.keys())):
        if key not in state_dict.keys():
            key_not_in_model_state_dict.append(key)
            del ckpt[key]
            continue
        if state_dict[key].size() != ckpt[key].size():
            del ckpt[key]
            size_mismatch_keys.append(key)

    # if(len(key_not_in_model_state_dict) != 0 or len(size_mismatch_keys) != 0):
    # print("⛳", end=" ")

    # print("==> Warning: The following key in the checkpoint is not presented in the model:", key_not_in_model_state_dict)
    # print("==> Warning: These keys have different size between checkpoint and current model: ", size_mismatch_keys)

    latent_diffusion.load_state_dict(ckpt, strict=False)

    trainer.validate(latent_diffusion, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        required=False,
        help="path to config .yaml file",
    )

    parser.add_argument(
        "--reload_from_ckpt",
        type=str,
        required=False,
        default=None,
        help="path to pretrained checkpoint",
    )

    parser.add_argument("--val", action="store_true")

    args = parser.parse_args()

    perform_validation = args.val

    assert torch.cuda.is_available(), "CUDA is not available"

    config_yaml = args.config_yaml

    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml_path = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)

    if args.reload_from_ckpt is not None:
        config_yaml["reload_from_ckpt"] = args.reload_from_ckpt

    if perform_validation:
        config_yaml["model"]["params"]["cond_stage_config"][
            "crossattn_audiomae_generated"
        ]["params"]["use_gt_mae_output"] = False
        config_yaml["step"]["limit_val_batches"] = None

    main(config_yaml, config_yaml_path, exp_group_name, exp_name, perform_validation)