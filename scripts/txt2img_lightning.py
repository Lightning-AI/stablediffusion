import argparse
import os
import time
import torch
from pytorch_lightning import seed_everything
from ldm.lightning import LightningStableDiffusion

def benchmark_fn(device, iters: int, warm_up_iters: int, function, *args, **kwargs) -> float:
    """
    Function for benchmarking a pytorch function.

    Parameters
    ----------
    iters: int
        Number of iterations.
    function: lambda function
        function to benchmark.
    args: Any type
        Args to function.
    Returns
    -------
    float
        Runtime per iteration in ms.
    """
    import torch

    results = []

    # Warm up
    for _ in range(warm_up_iters):
        function(*args, **kwargs)

    # Start benchmark.
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        torch.cuda.reset_peak_memory_stats()
    else:
        t0 = time.time()

    for _ in range(iters):
        results.extend(function(*args, **kwargs))

    if device == "cuda":
        max_memory = torch.cuda.max_memory_allocated(0)/2**20
        end_event.record()
        torch.cuda.synchronize()
        # in ms
        return (start_event.elapsed_time(end_event)) / iters, max_memory, results
    else:
        return (time.time() - t0) / iters, None, results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="astronaut riding a horse, digital art, epic lighting, highly-detailed masterpiece trending HQ",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./outputs"
    )
    parser.add_argument(
        "--sampler",
        default="ddim",
        help="default sampler",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=3,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--use_triton_attention",
        action='store_true',
        help="whether to use triton attention",
    )
    opt = parser.parse_args()
    return opt


def main(opt):
    opt = parse_args()
    os.makedirs(opt.outdir, exist_ok=True)
    seed_everything(opt.seed)

    device = "cuda" if torch.cuda.is_available() else "mps"

    steps = 30

    model = LightningStableDiffusion(
        config_path=opt.config,
        checkpoint_path=opt.ckpt,
        device=device,
        fp16=True, # Supported on GPU and CPU only, skipped otherwise.
        deepspeed=True, # Supported on Ampere and RTX, skipped otherwise.
        cuda_graph=True, # Currently enabled only for batch size 1.
        context="no_grad",
        flash_attention="hazy",
        steps=steps,
    )

    for batch_size in [1, 2, 4]:
        if batch_size == 1:
            t, max_memory, images = benchmark_fn(device, 10, 5, model.predict_step, prompts=opt.prompt, batch_idx=0)
        else:
            t, max_memory, images = benchmark_fn(device, 10, 5, model.predict_step, prompts=[opt.prompt] * batch_size, batch_idx=0)
        print(f"Average time {t} secs on batch size {batch_size}.")
        print(f"Max GPU Memory cost is {max_memory} MB.")

    grid_count = len(os.listdir(opt.outdir)) - 1

    for image in images:
        image.save(os.path.join(opt.outdir, f'grid-lightning-{opt.sampler}-{grid_count:04}.png'))
        grid_count += 1

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
