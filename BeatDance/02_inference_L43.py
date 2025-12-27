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

data_dir = "/data/han_data/dance_data/beatdance"
output_dir = "/data/han_data/dance_data/beatdance/models"
ann_dir = "/data/han_data/dance_data/pdl_chopped_2.csv"
os.makedirs(output_dir, exist_ok=True)
batch_size = 32
seglen = 2
L = 43
exp_name = f"seglen{seglen}_L{L}_batchsize{batch_size}_agg"
frame_rate = 25
sample_rate = 48000

def getitem(id, row):
    music_features_dir = os.path.join(data_dir, 'pdl_mert_s2_L43')
    pose_dir = os.path.join(data_dir, "pdl_pose")
    video_beats_dir = os.path.join(data_dir, 'pdl_vitpose_beats') # per video file
    music_beats_dir = os.path.join(data_dir, 'pdl_music_beats')  #per music file
    sao_rate = 21.5
    t_start = row["start"]
    t_end = row["end"]
    file_id = row["file_id"]
    music_feature_dir = os.path.join(music_features_dir, "{}.pt".format(id))
    music_avg_feature = torch.load(music_feature_dir)

    pose_path = os.path.join(pose_dir, "{}.pt".format(file_id))
    pose_feature = torch.load(pose_path) # postprocessed motion features (T, 138)
    pose_feature = pose_feature[int(frame_rate*t_start):int(frame_rate*t_end), :] #extract features (T, 138)

    data = {
            #'video': video_avg_feature, #(B,L,512)
            'music': music_avg_feature, #(B,L,768)
            'pose': pose_feature, # (B,T,138)
            'id':id,
        }
    video_beat_dir = os.path.join(video_beats_dir, "{}.pt".format(file_id))
    music_beat_dir = os.path.join(music_beats_dir, "{}.pt".format(file_id))
    video_beat = torch.load(video_beat_dir)
    music_beat = torch.load(music_beat_dir)
    
    # resample sampling rate of beats to 21.5 hz
    video_beat_avg_feature = beat_preprocess(video_beat, frame_rate, sao_rate, t_start, t_end) # (L, 1)
    music_beat_avg_feature = beat_preprocess(music_beat, sample_rate//512, sao_rate, t_start, t_end)  #(L, 4)

    data['video_beat'] = video_beat_avg_feature #(B,L,D)
    data['music_beat'] = music_beat_avg_feature #(B,L,D)
    return data

def beat_preprocess(beat_indices, fps, target_fps, start, end):
    frames = 43
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
    beats_vector = torch.zeros(int(frames*beat_dim))
    # resample the beat_indices (number of new samples / number of og samples)
    scale = frames * beat_dim / ((end-start) * fps)
    beat_indices = torch.round(beat_indices * scale).long()
    try:
        beats_vector[beat_indices] = 1 #(along beat_dim, every vector is the same)
    except:
        import pdb
        pdb.set_trace()
    return beats_vector.reshape((frames, int(beat_dim))) #reshape to (L, beat_dim)

def main():
    print("Current device: ", torch.cuda.get_device_name(0))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # limit to first gpu
    torch.multiprocessing.set_start_method('spawn')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = CusConfig("config/pdl_seg2_L43.json")
    config.exp_name = exp_name
    config.videos_dir = data_dir
    config.batch_size = batch_size
    config.num_frames = L
    config.output_dir = output_dir
    config.ann_dir = ann_dir
    config.tb_log_dir = os.path.join(output_dir, exp_name, 'logs')

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
    test_data_loader  = DataFactory.get_data_loader(config, split_type='test')
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
    
    trainer.load_checkpoint("model_best.pth")   
    # do inference and save the embeddings
    #import pdb
    
    trainer.model.eval()
    os.makedirs(os.path.join(data_dir, "fm_L43"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "fd_L43"), exist_ok=True)

    total_ids = 130800
    missing_ids = []
    for i in range(total_ids):
        if not os.path.exists(os.path.join(data_dir, "fm_L43", "{}.pt".format(i))):
            missing_ids.append(i)
    
    df = pd.read_csv(ann_dir)
    for iter, row in df.iterrows():
        if int(row["sample_id"]) in missing_ids:
            data = getitem(row["sample_id"], row)
            print("obtained features, ", row["sample_id"])
            for key in ['music', 'pose', 'music_beat', 'video_beat']:
                data[key] = data[key].unsqueeze(0).to(device)
            music_embed, video_embed = trainer.model(data) # EACH FEATURE WITHIN IS (B, L, D)

            torch.save(music_embed["music_fuse"][0,:,:], os.path.join(data_dir, "fm_L43", "{}.pt".format(int(row["sample_id"]))))
            torch.save(video_embed["video_fuse"][0,:,:], os.path.join(data_dir, "fd_L43", "{}.pt".format(int(row["sample_id"]))))
    
            #print(row["sample_id"])
    
    

    """
    with torch.no_grad(): # avoid storing gradients 
        for batch_idx, data in tqdm(enumerate(train_data_loader)): #EVERY DATA IS A BATCH OF FEATURES
            for key in ['music', 'pose', 'music_beat', 'video_beat']:
                data[key] = data[key].to(device)
            ids = data["id"]
            music_embed, video_embed = trainer.model(data) # EACH FEATURE WITHIN IS (B, L, D)
            for i, id in enumerate(ids):
                #pdb.set_trace()
                torch.save(music_embed["music_fuse"][i,:,:], os.path.join(data_dir, "fm_L43", "{}.pt".format(id)))
                torch.save(video_embed["video_fuse"][i,:,:], os.path.join(data_dir, "fd_L43", "{}.pt".format(id)))
        
        for batch_idx, data in tqdm(enumerate(test_data_loader)): #EVERY DATA IS A BATCH OF FEATURES
            for key in ['music', 'pose', 'music_beat', 'video_beat']:
                data[key] = data[key].to(device)
            ids = data["id"]
            music_embed, video_embed = trainer.model(data) # EACH FEATURE WITHIN IS (B, L, D)
            for i, id in enumerate(ids):
                torch.save(music_embed["music_fuse"][i,:,:], os.path.join(data_dir, "fm_L43", "{}.pt".format(id)))
                torch.save(video_embed["video_fuse"][i,:,:], os.path.join(data_dir, "fd_L43", "{}.pt".format(id)))
    
        for batch_idx, data in tqdm(enumerate(valid_data_loader)): #EVERY DATA IS A BATCH OF FEATURES
            for key in ['music', 'pose', 'music_beat', 'video_beat']:
                data[key] = data[key].to(device)
            ids = data["id"]
            music_embed, video_embed = trainer.model(data) # EACH FEATURE WITHIN IS (B, L, D)
            for i, id in enumerate(ids):
                torch.save(music_embed["music_fuse"][i,:,:], os.path.join(data_dir, "fm_L43", "{}.pt".format(id)))
                torch.save(video_embed["video_fuse"][i,:,:], os.path.join(data_dir, "fd_L43", "{}.pt".format(id)))
    """


if __name__ == '__main__':
    main()
