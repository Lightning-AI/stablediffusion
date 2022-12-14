from typing import List
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from omegaconf import OmegaConf
import lightning as L
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import tarfile
from PIL import Image


class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        super().__init__()
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, i: int) -> str:
        return self.prompts[i]


class LightningStableDiffusion(L.LightningModule):
    def __init__(
        self,
        device: torch.device,
        config_path: str = "configuration.yaml",
        checkpoint_path: str =  "checkpoint.ckpt",
        size: int = 512,
    ):
        super().__init__()

        config = OmegaConf.load(f"{config_path}")
        config.model.params.unet_config["params"]["use_fp16"] = False
        config.model.params.cond_stage_config["params"] = {"device": device}

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(state_dict, strict=False)

        self.sampler = DDIMSampler(self.model)

        self.initial_size = int(size / 8)
        self.steps = 50

        self.to(device)

    @torch.no_grad()
    def predict_step(self, prompts: List[str], batch_idx: int):
        batch_size = len(prompts)

        with self.model.ema_scope():
            uc = self.model.get_learned_conditioning(batch_size * [""])
            c = self.model.get_learned_conditioning(prompts)
            shape = [4, self.initial_size, self.initial_size]
            samples_ddim, _ = self.sampler.sample(
                S=self.steps,  # Number of inference steps, more steps -> higher quality
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=9.0,
                unconditional_conditioning=uc,
                eta=0.0,
            )

            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

            x_samples_ddim = (255.0 * x_samples_ddim).astype(np.uint8)
            pil_results = [Image.fromarray(x_sample) for x_sample in x_samples_ddim]
        return pil_results



def get_weights_version(version):

        if version == 1.5:
            os.system(
                "curl -C - https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml -o configuration.yaml"
            )

            os.system(
                "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/v1-5-pruned-emaonly.ckpt -o checkpoint.ckpt"
            )

        elif version == 1.4:
            os.system(
                "curl -C - https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml -o configuration.yaml"
            )

            os.system(
                "curl -C -  https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/sd_weights.tar.gz -o checkpoint.tar.gz"
            )
            file = tarfile.open('checkpoint.tar.gz')
            file.extractall()
            file.close()
        
        elif version == 2.0:
            os.system(
                "curl -C - https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml -o configuration.yaml"
            )

            os.system(
                "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/v1-5-pruned-emaonly.ckpt -o checkpoint.ckpt"
            )
        elif version == 2.1:
            os.system(
                "curl -C - https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml -o configuration.yaml"
            )

            os.system(
                "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/v1-5-pruned-emaonly.ckpt -o checkpoint.ckpt"
            )
        else:
            raise Exception("Error: Wrong version. Available versions: 1.4, 1.5, 2.0, and 2.1. Please ensure that you are using the correct version.")
   