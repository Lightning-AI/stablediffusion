from typing import List, Tuple, Union, Optional, Literal
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
from copy import deepcopy
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
        deepspeed: bool = False,
        cuda_graph: bool = False,
        flash_attention: Optional[Literal['hazy', 'triton']] = None,
        context: Optional[Literal['inference_mode', 'no_grad']] = None,
        # depreceated arguments
        use_deepspeed: bool = None,

    ):
        super().__init__()

        if use_deepspeed is not None:
            logger.warn(f"The argument `use_deepspeed` is depreceated and will be removed soon. Use `deepspeed={use_deepspeed}` instead.")
            deepspeed = use_deepspeed

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

        if deepspeed or cuda_graph or flash_attention:
            deepspeed_injection(
                self.model,
                fp16=fp16,
                cuda_graph=cuda_graph,
                flash_attention=flash_attention
            )

        # Replace with 
        self.sampler = _SAMPLERS[sampler](self.model)

        self.initial_size = int(size / 8)
        self.steps = steps or _STEPS[sampler]

        self.to(device, dtype=torch.float16 if fp16 else torch.float32)
        self.fp16 = fp16
        self.context = context

    def predict_step(self, prompts: Union[List[str], str], batch_idx: int = 0):
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)

        precision_scope = autocast if self.fp16 else nullcontext
        inference = torch.inference_mode if self.context == "inference_mode" else torch.no_grad
        inference = inference if torch.cuda.is_available() else nullcontext
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

    def in_loop_predict_step(
        self, inputs, total_steps=30, eta=0, unconditional_guidance_scale=7.5, x_T=None, ddim_use_original_steps=False,
        callback=None, timesteps=None, quantize_denoised=False,
        mask=None, x0=None, img_callback=None, log_every_t=100,
        temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
        unconditional_conditioning=None, dynamic_threshold=None,
        ucg_schedule=None
    ):
        # Collect the current hashes
        hashes = [v for v in list(inputs) if v != "global_state"]

        # Create the global state to keep track of the batched img, etc..
        if "global_state" not in inputs:
            inputs["global_state"] = {}

        inputs["global_state"]['hashes'] = hashes

        if len(hashes) == 0:
            return

        # Create the context managers
        inference = torch.inference_mode if self.context == "inference_mode" else torch.no_grad
        inference = inference if torch.cuda.is_available() else nullcontext
        precision_scope = autocast if self.fp16 else nullcontext

        bs = len(hashes)

        with inference():
            with precision_scope("cuda"):

                # Setup the unconditional_conditioning tensor and sampler.
                if not hasattr(self, "shape"):
                    # To be cached
                    self.shape = [4, self.initial_size, self.initial_size]
                    self.unconditional_conditioning = self.model.get_learned_conditioning([""])
                    self.sampler.make_schedule(ddim_num_steps=total_steps, ddim_eta=eta, verbose=False)
                    timesteps = self.sampler.ddim_timesteps
                    self.time_range = torch.tensor(np.flip(timesteps).tolist(), device=self.device)

                # Pre-processing
                indexed_prompts = [(index, value) for index, value in inputs.items() if isinstance(value, str)]

                # Detect if there were an exit without new prompts.
                if 'previous_hashes' in inputs["global_state"]:
                    changed_hash = inputs["global_state"]['previous_hashes'] != inputs["global_state"]['hashes']
                else:
                    changed_hash = False

                # New prompts are passed.
                if indexed_prompts or changed_hash:

                    if indexed_prompts:
                        conditioning = self.model.get_learned_conditioning([e[1] for e in indexed_prompts])
                        conditioning_unbind = torch.unbind(conditioning)
                        for idx, (index, _) in enumerate(indexed_prompts):
                            if "step" not in inputs[index]:
                                inputs[index] = {
                                    "conditioning": conditioning_unbind[idx].unsqueeze(0),
                                    "step": 0,
                                    "img": torch.randn((1, *self.shape), device=self.device)
                                }

                    # Unpack if new prompt are added changed
                    if "img" in inputs["global_state"]:
                        imgs = torch.unbind(inputs["global_state"]['img'])
                        steps = torch.unbind(inputs["global_state"]['steps'])

                        for index, hash in enumerate(inputs["global_state"]['previous_hashes']):
                            inputs[hash]['img'] = imgs[index].unsqueeze(0)
                            inputs[hash]['step'] = steps[index]

                        del inputs["global_state"]['img']

                    # Re-create the tensors
                    if "img" not in inputs["global_state"]:
                        # Concat the img and conditioning
                        inputs["global_state"]['img'] = torch.cat([v['img'] for k, v in inputs.items() if k != "global_state"]).contiguous()
                        inputs["global_state"]['conditioning'] = torch.cat([v['conditioning'] for k, v in inputs.items() if k != "global_state"]).contiguous()
                        inputs["global_state"]['unconditional_conditioning'] = self.unconditional_conditioning.repeat(bs, 1, 1).contiguous()

                        # Generate the steps and stack them
                        steps = []
                        for hash in hashes:
                            v = inputs[hash]
                            step = v['step']
                            tensor = torch.tensor(step, dtype=torch.long)
                            steps.append(tensor)

                        inputs["global_state"]['steps'] = torch.stack(steps)

                # Generate the alphas index and timesteps
                inputs["global_state"]['index'] = total_steps - inputs["global_state"]['steps']
                inputs["global_state"]['ts'] = self.time_range[inputs["global_state"]['steps']]

                results = {}

                # print(inputs["global_state"]['img'].shape)
                # print(inputs["global_state"]['conditioning'].shape)
                # print(inputs["global_state"]['unconditional_conditioning'].shape)

                # Make a step with the sampler
                img, _ = self.sampler.p_sample_ddim(
                    inputs["global_state"]['img'], inputs["global_state"]['conditioning'],
                    inputs["global_state"]['ts'], index=inputs["global_state"]['index'], use_original_steps=False,
                    quantize_denoised=False, temperature=temperature,
                    noise_dropout=noise_dropout, score_corrector=score_corrector,
                    corrector_kwargs=corrector_kwargs,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=inputs["global_state"]['unconditional_conditioning'],
                    dynamic_threshold=dynamic_threshold
                )
                
                # Store the result
                inputs["global_state"]['img'] = img

                # Increment the step
                inputs["global_state"]['steps'] += 1

                print(inputs["global_state"]['steps'])

                # Detect if any tensor needs to be extracted.
                matches = inputs["global_state"]['steps'] == total_steps
                if any(matches):
                    imgs = torch.unbind(inputs["global_state"]['img'])
                    steps = torch.unbind(inputs["global_state"]['steps'])
                    for index, match in enumerate(matches):
                        hash = hashes[index]
                        img = imgs[index].unsqueeze(0)
                        step = steps[index]
                        if match:
                            img = self.model.decode_first_stage(img)
                            img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
                            img = img.cpu().permute(0, 2, 3, 1).numpy()
                            img = (255.0 * img).astype(np.uint8)
                            results[hash] = Image.fromarray(img[0])
                            del inputs[hash]
                        else:
                            inputs[hash]['img'] = img
                            inputs[hash]['step'] = step
                    del inputs["global_state"]['img']
                    inputs["global_state"]['previous_hashes'] = [k for k, m in zip(inputs["global_state"]['hashes'], matches) if not m]
                else:
                    inputs["global_state"]['previous_hashes'] = deepcopy(inputs["global_state"]['hashes'])
        return results


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