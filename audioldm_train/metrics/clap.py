import laion_clap
import torch
import torch.nn.functional as F


class CLAPScoreCalculator:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clapmodel = laion_clap.CLAP_Module(enable_fusion=True, device=device)
        self.clapmodel.load_ckpt()
        self.clapmodel.model.audio_branch.requires_grad_(False)
        self.clapmodel.model.audio_branch.eval()

    def calculate(self, audio1, audio2):
        if audio1.ndim == 2:
            audio1 = torch.mean(audio1, dim=0)
        if audio2.ndim == 2:
            audio2 = torch.mean(audio2, dim=0)
        with torch.cuda.amp.autocast(enabled=False):
            embed1 = self.clapmodel.get_audio_embedding_from_data(
                audio1.float()[None, :], use_tensor=True
            )
            embed2 = self.clapmodel.get_audio_embedding_from_data(
                audio2.float()[None, :], use_tensor=True
            )
        return F.cosine_similarity(embed1, embed2, dim=-1).item()
