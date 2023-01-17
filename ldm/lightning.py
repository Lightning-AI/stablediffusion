from typing import List, Tuple, Union, Optional
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
import base64
from omegaconf import OmegaConf
import lightning as L
import torch
from io import BytesIO
from torch.utils.data import Dataset
import numpy as np
from time import time
from PIL import Image
from io import BytesIO
from contextlib import nullcontext
from torch import autocast
from ldm.deepspeed_replace import deepspeed_injection, ReplayCudaGraphUnet
import logging

logger = logging.getLogger(__name__)


class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        super().__init__()
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, i: int) -> str:
        return self.prompts[i]

_SAMPLERS = {
    "ddim": DDIMSampler,
    "plms": PLMSSampler,
    "dpm": DPMSolverSampler
}

_STEPS = {
    "ddim": 50,
    "plms": 50,
    "dpm": 50
}


class LightningStableDiffusion(L.LightningModule):
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str,
        size: int = 512,
        fp16: bool = True,
        sampler: str = "ddim",
        steps: Optional[int] = None,
        use_deepspeed: bool = False,
        enable_cuda_graph: bool = False,
        use_triton_attention: bool = True,
        use_inference_context: bool = False,
    ):
        super().__init__()

        if device in ("mps", "cpu") and fp16:
            logger.warn(f"You provided fp16=True but it isn't supported on `{device}`. Skipping...")
            fp16 = False

        config = OmegaConf.load(f"{config_path}")
        config.model.params.unet_config["params"]["use_fp16"] = False
        config.model.params.cond_stage_config["params"] = {"device": device}
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(state_dict, strict=False)

        if use_deepspeed or enable_cuda_graph or use_triton_attention:
            deepspeed_injection(
                self.model,
                fp16=fp16,
                enable_cuda_graph=enable_cuda_graph,
                use_triton_attention=use_triton_attention
            )

        # Replace with 
        self.sampler = _SAMPLERS[sampler](self.model)

        self.initial_size = int(size / 8)
        self.steps = steps or _STEPS[sampler]

        self.to(device, dtype=torch.float16 if fp16 else torch.float32)
        self.fp16 = fp16
        self.use_inference_context = use_inference_context

    def predict_step(self, prompts: Union[List[str], str], batch_idx: int = 0):
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)

        precision_scope = autocast if self.fp16 else nullcontext
        inference = torch.inference_mode if torch.cuda.is_available() else torch.no_grad
        inference = inference if self.use_inference_context else nullcontext
        with inference():
            with precision_scope("cuda"):
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
                        unconditional_guidance_scale=7.5,
                        unconditional_conditioning=uc,
                        eta=0.0,
                    )

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_samples_ddim = (255.0 * x_samples_ddim).astype(np.uint8)
                    pil_results = [Image.fromarray(x_sample) for x_sample in x_samples_ddim]
        return pil_results


class LightningStableImg2ImgDiffusion(L.LightningModule):
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str,
        size: int = 512,
    ):
        super().__init__()

        config = OmegaConf.load(config_path)
        pl_sd = torch.load(checkpoint_path, map_location="cpu")
        sd = pl_sd["state_dict"]
        self.model = instantiate_from_config(config.model).to(device)
        self.model.load_state_dict(sd, strict=False)

        # Update Unet for inference
        # Currently waiting for https://github.com/pytorch/pytorch/issues/91302
        self.model.model = ReplayCudaGraphUnet(self.model.model)

        self.to(device, dtype=torch.float16)

        self.sampler = DDIMSampler(self.model)

        self.initial_size = int(size / 8)
        self.steps = 50

        self._device = device

    def serialize_image(self, image: str):
        init_image = base64.b64decode(image)
        buffer = BytesIO(init_image)
        init_image = Image.open(buffer, mode="r").convert("RGB")
        image = init_image.resize((512, 512), resample=Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2. * image - 1.

    @torch.inference_mode()
    def predict_step(
        self,
        inputs: Tuple[Union[str, List[str]], Union[str, List[str]]],
        batch_idx: int,
        precision=16,
        strength=0.75, 
        scale = 5.0
    ):
        t0 = time()

        prompt, init_image = inputs

        if isinstance(init_image, str):
            init_image = [init_image]

        if isinstance(prompt, str):
            prompt = [prompt]

        assert len(prompt) == len(init_image)

        batch_size = len(init_image)

        precision_scope = autocast if precision == 16 else nullcontext
        with precision_scope("cuda"):
            init_image = torch.cat([self.serialize_image(img).to(self._device, dtype=torch.float16) for img in init_image], dim=0)
            init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))

            self.sampler.make_schedule(ddim_num_steps=self.steps, ddim_eta=0.0, verbose=False)

            t_enc = int(strength * self.steps)

            uc = None
            if scale != 1.0:
                uc = self.model.get_learned_conditioning(batch_size * [""])
            c = self.model.get_learned_conditioning(prompt)

            # encode (scaled latent)
            z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(self._device))
            # decode it
            samples = self.sampler.decode(
                z_enc,
                c,
                t_enc,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
            )

            x_samples_ddim = self.model.decode_first_stage(samples)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

            x_samples_ddim = (255.0 * x_samples_ddim).astype(np.uint8)
            pil_results = [Image.fromarray(x_sample) for x_sample in x_samples_ddim]

        print(f"Generated {batch_size} images in {time() - t0}")
        return pil_results