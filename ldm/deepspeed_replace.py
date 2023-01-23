'''
Credits to The Microsoft DeepSpeed Team
'''

import torch
from functools import partial
from dataclasses import dataclass
import time
from lightning_utilities.core.imports import package_available
from ldm.modules.attention import CrossAttention, BasicTransformerBlock
from ldm.models.diffusion.ddpm import DiffusionWrapper
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from ldm.detect_target import _detect_cuda
import logging
import math
from torch import nn, einsum
from einops import rearrange

# TODO: Remove once resolved: https://github.com/microsoft/DeepSpeed/issues/2715
torch.nn.Paramaeter = torch.nn.Parameter

if package_available("deepspeed"):
    import deepspeed.ops.transformer as transformer_inference
    from deepspeed.ops.transformer.inference.diffusers_attention import DeepSpeedDiffusersAttention
    from deepspeed.ops.transformer.inference.diffusers_transformer_block import DeepSpeedDiffusersTransformerBlock
    from deepspeed.ops.transformer.inference.diffusers_2d_transformer import Diffusers2DTransformerConfig
    from deepspeed.inference.engine import InferenceEngine
    from deepspeed.module_inject.replace_policy import UNetPolicy, DSPolicy
else:
    class InferenceEngine:
        pass

    class DSPolicy:
        pass

    class DeepSpeedDiffusersTransformerBlock:
        pass

logger = logging.getLogger(__name__)

class InferenceEngine(InferenceEngine):

    def __init__(self, *args, cuda_graph_global: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_graph_global = cuda_graph_global

    def forward(self, *inputs, **kwargs):
        """Execute forward propagation

        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """
        start = None
        if self.model_profile_enabled and self.cuda_graph_global:
            torch.cuda.synchronize()
            start = time.time()

        if self.cuda_graph_global:
            if self.cuda_graph_created:
                outputs = self._graph_replay(*inputs, **kwargs)
            else:
                self._create_cuda_graph(*inputs, **kwargs)
                outputs = self._graph_replay(*inputs, **kwargs)
        else:
            outputs = self.module(*inputs, **kwargs)

        if self.model_profile_enabled and self.cuda_graph_global:
            torch.cuda.synchronize()
            duration = time.time() - start
            self._model_times.append(duration)

        return outputs


@dataclass
class CudaGraphRecord:

    graph = None
    args = None
    kwargs = None
    output = None
    cuda_graph_created: bool = False
    cuda_graph: bool = True


class CudaGraphBatchRecord(dict):

    def __init__(self, cuda_graph):
        super().__init__()
        self.cuda_graph = cuda_graph

    def __getitem__(self, key):
        if key not in self:
            self[key] = CudaGraphRecord(cuda_graph=self.cuda_graph)
        for item, value in self.items():
            if item == key:
                return value
        raise Exception()


class CudaGraphInferenceModule(torch.nn.Module):

    # Inspired from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/model_implementations/diffusers/vae.py

    inference_methods = ["forward"]

    def __init__(self, module, cuda_graph = True, batch_sizes=[1]):
        super().__init__()
        self.module = module
        self.module.requires_grad_(requires_grad=False)
        self.module.to(memory_format=torch.channels_last)
        self.cuda_graph_records = {}
        self.batch_sizes = batch_sizes

        for method_name in self.inference_methods:
            fn = getattr(self, f"_{method_name}")
            assert fn
            self.cuda_graph_records[method_name] = CudaGraphBatchRecord(cuda_graph=cuda_graph)
            setattr(self, method_name, partial(self._apply_fn, fn=fn, graph_record=self.cuda_graph_records[method_name]))

    def __getattr__(self, key):
        if hasattr(self._modules["module"], key) and key not in self.__dict__:
            return getattr(self._modules["module"], key)
        if key == "module":
            return self._modules["module"]
        return object.__getattribute__(self, key)

    def _graph_replay(self, graph_record, *args, **kwargs):
        for i in range(len(args)):
            if torch.is_tensor(args[i]):
                graph_record.args[i].copy_(args[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                graph_record.kwargs[k].copy_(kwargs[k])
        graph_record.graph.replay()
        return graph_record.output

    def extract_batch_size(self, *args, **kwargs) -> int:
        raise NotImplementedError

    def _apply_fn(self, *args, fn=None, graph_record=None,  **kwargs):
        batch_size = self.extract_batch_size(*args, **kwargs)
        if batch_size in self.batch_sizes and graph_record[batch_size].cuda_graph:
            if graph_record[batch_size].cuda_graph_created:
                outputs = self._graph_replay(graph_record[batch_size], *args, **kwargs)
            else:
                self._create_cuda_graph(fn, graph_record[batch_size], *args, **kwargs)
                outputs = self._graph_replay(graph_record[batch_size], *args, **kwargs)
            return outputs
        return fn(*args, **kwargs)

    def _create_cuda_graph(self, fn, graph_record, *args, **kwargs):
        # Warmup to create the workspace and cublas handle
        cuda_stream = torch.cuda.Stream()
        cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cuda_stream):
            for _ in range(3):
                fn(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(cuda_stream)

        # Capture inputs to the graph
        graph_record.graph = torch.cuda.CUDAGraph()
        graph_record.args = args
        graph_record.kwargs = kwargs

        with torch.cuda.graph(graph_record.graph):
            # Store output
            graph_record.output = fn(*graph_record.args, **graph_record.kwargs)

        graph_record.cuda_graph_created = True


class ReplayCudaGraphUnet(CudaGraphInferenceModule):

    def extract_batch_size(self, sample, timestamp, c_crossattn) -> int:
        return sample.shape[0] // 2

    def _forward(self, sample, timestamp, c_crossattn):
        return self.module(sample, timestamp, c_crossattn=c_crossattn)


class ReplayCudaGraphVAE(CudaGraphInferenceModule):

    inference_methods = ["forward", "encode", "decode"]

    def extract_batch_size(self, input, **__) -> int:
        return input.shape[0]

    def _encode(self, x):
        return self.module.encode(x)

    def _decode(self, x):
        return self.module.decode(x)

    def _forward(self, input, sample_posterior=True):
        return self.module(input, sample_posterior=sample_posterior)


class ReplayCudaGraphClipEncoder(CudaGraphInferenceModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module.transformer.text_model._build_causal_attention_mask = self._build_causal_attention_mask

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        mask = torch.empty(bsz,
                           seq_len,
                           seq_len,
                           dtype=dtype,
                           device=torch.cuda.current_device())
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)
        mask = mask.unsqueeze(1)
        return mask

    def extract_batch_size(self, sample, **__) -> int:
        return sample.shape[0]

    def _forward(self, *inputs, **kwargs):
        return self.enc(*inputs, **kwargs)


class UNetPolicy(DSPolicy):

    def match(self, module):
        return isinstance(module, DiffusionWrapper)

    def apply(self, module, cuda_graph=True):
        if cuda_graph:
            return ReplayCudaGraphUnet(module, cuda_graph=cuda_graph)
        return module

    def attention(self, client_module, flash_attention: str = "triton"):
        qw = client_module.to_q.weight
        kw = client_module.to_k.weight
        vw = client_module.to_v.weight

        if flash_attention == "triton" and qw.shape[1] == kw.shape[1]:
            qkvw = torch.nn.Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)

            return qkvw, \
                   client_module.to_out[0].weight, \
                   client_module.to_out[0].bias, \
                   qw.shape[-1], \
                   client_module.heads
        else:
            return qw, \
                   kw, vw, \
                   client_module.to_out[0].weight, \
                   client_module.to_out[0].bias, \
                   qw.shape[-1], \
                   client_module.heads


class VAEPolicy(DSPolicy):

    def match(self, module):
        return isinstance(module,  AutoencoderKL)

    def apply(self, module, cuda_graph=True):
        if cuda_graph:
            return ReplayCudaGraphVAE(module, cuda_graph=cuda_graph)
        return module


class ClipEncoderPolicy(DSPolicy):

    def match(self, module):
        return isinstance(module, FrozenCLIPEmbedder)

    def apply(self, module, cuda_graph=True):
        if cuda_graph:
            return ReplayCudaGraphClipEncoder(module, cuda_graph=cuda_graph)
        return module


GENERIC_POLICIES = [UNetPolicy, VAEPolicy, ClipEncoderPolicy]



def _module_match(module):
    for policy in GENERIC_POLICIES:
        policy = policy()
        if policy.match(module):
            return policy
    return None


class FlashAttention(nn.Module):

    """
    Use https://github.com/HazyResearch/flash-attention Flash Attention Kernels.
    
    Note: The Flash Attention kernels are activate only when the head dimension is lower than 128 and no context tensor is provided. 
    """

    def __init__(self, hidden_size: int, hidden_size_out: int, heads: int, fp16: bool = True, device: str = "cuda", max_out_tokens: int = 4096):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_out_tokens = max_out_tokens
        self.heads = heads

        dtype = torch.float16 if fp16 else torch.float32

        self.attn_kw = nn.Parameter(
            torch.empty(
                hidden_size,
                hidden_size,
                dtype=dtype,
                device=device
            ), requires_grad=False
        )

        self.attn_vw = nn.Parameter(
            torch.empty(
                hidden_size,
                hidden_size,
                dtype=dtype,
                device=device
            ), requires_grad=False)

        self.attn_qw = nn.Parameter(
            torch.empty(
                hidden_size,
                hidden_size_out,
                dtype=dtype,
                device=device
            ), requires_grad=False
        )

        self.attn_ow = nn.Parameter(torch.empty(hidden_size,
                                                hidden_size,
                                                dtype=dtype,
                                                device=device),
                                    requires_grad=False)

        self.attn_ob = nn.Parameter(torch.empty(hidden_size,
                                                dtype=dtype,
                                                device=device),
                                    requires_grad=False)
        self.do_out_bias = True

        self.norm_factor = math.sqrt(math.sqrt(hidden_size // heads))

        self.scale = (1 / self.norm_factor) * (1 / self.norm_factor) 

    def forward(self, input, context=None, input_mask=None):
        from flash_attn.flash_attn_interface import _flash_attn_forward

        batch_size = input.shape[0]
        seqlen = input.shape[1]
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=self.attn_kw.device)
        do_flash_attn = (input.shape[-1] / self.heads <= 128) and context is None

        if not do_flash_attn:
            query = torch.matmul(input, self.attn_qw)
            if context is None:
                key = torch.matmul(input, self.attn_kw)
                value = torch.matmul(input, self.attn_vw)
            else:
                key = torch.matmul(context, self.attn_kw)
                value = torch.matmul(context, self.attn_vw)

            query = rearrange(query, 'b n (h d) -> (b h) n d', h=self.heads)
            key = rearrange(key, 'b n (h d) -> (b h) n d', h=self.heads)
            value = rearrange(value, 'b n (h d) -> (b h) n d', h=self.heads)
        else:
            query = torch.matmul(input, self.attn_qw)
            key = torch.matmul(input, self.attn_kw)
            value = torch.matmul(input, self.attn_vw)

            query = query.reshape((batch_size * seqlen, self.heads, -1)).contiguous()
            key = key.reshape((batch_size * seqlen, self.heads, -1)).contiguous()
            value = value.reshape((batch_size * seqlen, self.heads, -1)).contiguous()

        if do_flash_attn:
            context_layer, _, _ = _flash_attn_forward(query, key, value, torch.empty_like(query), cu_seqlens, cu_seqlens, seqlen, seqlen, 0.0, self.scale, causal=None, return_softmax=False)
            context_layer = context_layer.reshape((batch_size, seqlen, -1))
        else:
            sim = einsum('b i d, b j d -> b i j', query, key) * self.scale
            del query, key

            sim = sim.softmax(dim=-1)

            out = einsum('b i j, b j d -> b i d', sim, value)
            context_layer = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)

        return torch.matmul(context_layer, self.attn_ow)


class DeepSpeedFlashDiffusersTransformerBlock(DeepSpeedDiffusersTransformerBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.attn_1, FlashAttention):
            self.attn_1.do_out_bias = False
            self.attn_1_bias = self.attn_1.attn_ob

        if isinstance(self.attn_2, FlashAttention):
            self.attn_2.do_out_bias = False
            self.attn_2_bias = self.attn_2.attn_ob


# Inspired from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_module.py#L201
def deepspeed_injection(
    module,
    fp16=True,
    cuda_graph=True,
    flash_attention=None
):

    add_warning = False

    if not torch.cuda.is_available():
        logger.warn("You provided deepspeed=True but Deepspeed isn't supported on your architecture. Skipping...")
        return

    def replace_attn(child, policy):
        nonlocal add_warning

        if flash_attention is None:
            return child

        policy_attn = policy.attention(child, flash_attention=flash_attention)
        
        if policy_attn is None:
            return child

        if len(policy_attn) == 5:
            qkvw, attn_ow, attn_ob, hidden_size, heads = policy_attn
        else:
            qw, kw, vw, attn_ow, attn_ob, hidden_size, heads = policy_attn

        if flash_attention == "triton":
            if not add_warning:
                logger.warn("Using DeepSpeed Triton Flash Attention. Skipped if a context is provided or with dim head higher than 128.")
                add_warning = True
            config = transformer_inference.DeepSpeedInferenceConfig(
                hidden_size=hidden_size,
                heads=heads,
                fp16=fp16,
                triangular_masking=False,
                max_out_tokens=4096,
            )
            attn_module = DeepSpeedDiffusersAttention(config)
        else:
            if not add_warning:
                logger.warn("Using Flash Attention. Skipped if a context is provided or with dim head higher than 128.")
                add_warning = True
            attn_module = FlashAttention(
                hidden_size=hidden_size,
                heads=heads,
                fp16=fp16,
                hidden_size_out=hidden_size,
                max_out_tokens=4096,
            )                

        def transpose(data):
            data = data.contiguous()
            data.reshape(-1).copy_(data.transpose(-1, -2).contiguous().reshape(-1))
            data = data.reshape(data.shape[-1], data.shape[-2])
            data.to(torch.cuda.current_device())
            return data

        if len(policy_attn) == 5:
            attn_module.attn_qkvw.data = transpose(qkvw.data)
        else:
            attn_module.attn_qkvw = None
            attn_module.attn_qw.data = transpose(qw.data)
            attn_module.attn_kw.data = transpose(kw.data)
            attn_module.attn_vw.data = transpose(vw.data)

        attn_module.attn_qkvb = None
        attn_module.attn_ow.data = transpose(attn_ow.data)
        attn_module.attn_ob.data.copy_(attn_ob.data.to(torch.cuda.current_device()))
        return attn_module

    def replace_attn_block(child, policy):
        # Track DeepSpeed Issue: https://github.com/microsoft/DeepSpeed/issues/2681
        config = Diffusers2DTransformerConfig(int8_quantization=False)
        return DeepSpeedFlashDiffusersTransformerBlock(child, config)

    new_policies = {
        CrossAttention: replace_attn,
        BasicTransformerBlock: replace_attn_block,
    }

    for name, sub_module in module.named_children():

        policy = _module_match(sub_module)

        if policy is None:
            continue

        def _replace_module(module, policy):
            for name, child in module.named_children():
                _replace_module(child, policy)
                if child.__class__ in new_policies:
                    replaced_module = new_policies[child.__class__](child, policy)
                    setattr(module, name, replaced_module)

        if not package_available("deepspeed"):
            logger.warn("You provided deepspeed=True but Deepspeed isn't installed. Skipping...")
        elif flash_attention == "triton" and _detect_cuda() not in ["80"]:
            logger.warn("You provided deepspeed=True but Deepspeed isn't supported on your architecture. Skipping...")
        else:
            _replace_module(sub_module, policy)

        new_module = policy.apply(sub_module, cuda_graph=cuda_graph)
        print(f"**** found and replaced {name} w. {type(new_module)}")
        setattr(module, name, new_module)

