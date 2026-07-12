"""Shared checkpoint loading helpers for ControlNet train and evaluation CLIs."""

from huggingface_hub import hf_hub_download


def download_checkpoint(checkpoint_name="audioldm2-full"):
    if "audioldm2-speech" in checkpoint_name:
        model_id = "haoheliu/audioldm2-speech"
    else:
        model_id = f"haoheliu/{checkpoint_name}"
    return hf_hub_download(
        repo_id=model_id,
        filename=f"{checkpoint_name}.pth",
    )


def modify_state_dict(state_dict, modify_dict):
    for target_key, modification in modify_dict.items():
        if target_key not in state_dict:
            continue
        state_dict[modification["new_key"]] = state_dict[target_key]
        if not modification["duplicate"]:
            del state_dict[target_key]
    return state_dict


def filter_compatible_state_dict(checkpoint, model_state_dict):
    """Remove checkpoint entries absent from, or incompatible with, a model."""
    missing_keys = []
    size_mismatch_keys = []
    for key in list(checkpoint):
        if key not in model_state_dict:
            missing_keys.append(key)
            del checkpoint[key]
        elif model_state_dict[key].shape != checkpoint[key].shape:
            size_mismatch_keys.append(key)
            del checkpoint[key]
    return checkpoint, missing_keys, size_mismatch_keys
