export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/sangheon/Desktop/AudioLDM-ControlNet:$PYTHONPATH
python3 audioldm_train/train/latent_diffusion_controlnet_beatdance.py -c audioldm_train/config/2025_11_23_dance_controlnet_beatdance/audioldm_original_medium.yaml --reload_from_ckpt data/checkpoints/audioldm-m-full.ckpt