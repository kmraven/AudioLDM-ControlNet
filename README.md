# AudioLDM-Dance-ControlNet
## How to run training
1. build env with `environment.yaml`

2. download checkpoints and put them into `data/checkpoints`
- AudioLDM related stuffs: follow the instruction in [original AudioLDM README](https://github.com/haoheliu/AudioLDM-training-finetuning?tab=readme-ov-file#download-checkpoints-and-dataset)
- AudioLDM model: follow the instruction in [original AudioLDM README](https://github.com/haoheliu/AudioLDM-training-finetuning?tab=readme-ov-file#finetuning-of-the-pretrained-model)
- MotionBERT: donwload the 'MotionBERT (162MB)' ckpt from [original MotionBERT README](https://github.com/Walter0807/MotionBERT?tab=readme-ov-file#model-zoo) and put them into `data/checkpoints`

3. put the aist++ dataset that is provided by `Textual inversion repo` ([link](https://github.com/lsfhuihuiff/Dance-to-music_Siggraph_Asia_2024?tab=readme-ov-file#prepare-dataset))

4. put the BeatDance your config somewhere, then write the path into line 188 of `audioldm_train/config/2025_11_23_dance_controlnet_beatdance/audioldm_original_medium.yaml`
Called params in this AudioLDM-ControlNet is only model-related params. Please check `audioldm_train/modules/motion_encoder/BeatDance/model/clip_transformer.py`

5. start training  
```
% sh ./bash_train_controlnet_beatdance.sh
```