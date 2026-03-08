"Largely taken and adapted from https://github.com/lucidrains/video-diffusion-pytorch"

import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T
from torch.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from ddpm.text import BERT_MODEL_DIM

from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL

import matplotlib.pyplot as plt
import numpy as np
import cv2
# MONAI Generative models
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from params import *
from ddpm.lora import inject_trainable_lora

# Wandb for experiment tracking
import wandb

# Metrics for validation
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Function to convert a color image tensor to grayscale using PIL
def to_grayscale(tensor):
    # Convert PyTorch tensor to numpy array
    np_img = tensor.cpu().numpy()

    # Convert numpy array to PIL Image (assuming the input is RGB)
    pil_img = Image.fromarray(np.transpose(np_img, (1, 2, 0)).astype(np.uint8), mode='RGB')

    # Convert PIL Image to grayscale
    pil_img_gray = pil_img.convert('L')

    # Optionally, convert back to PyTorch tensor
    gray_tensor = torch.from_numpy(np.array(pil_img_gray)).unsqueeze(0)  # Add channel dimension

    return gray_tensor


# helpers functions
def make_rgb(volume):
    """Tile a NumPy array to make sure it has 3 channels."""
    z, c, h, w = volume.shape

    tiling_shape = [1]*(len(volume.shape))
    tiling_shape[1] = 3
    np_vol = torch.tile(volume, tiling_shape)
    return np_vol


def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias
class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)

# attention along space and time
class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias=None,
        focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# model
class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type='resnet',
        resnet_groups=8,
        medclip = True,
        classifier_free_guidance = False
    ):
        super().__init__()
        self.channels = channels
        self.dim_mults = dim_mults
        self.medclip = medclip
        self.cfg = classifier_free_guidance

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head)) # 32

        def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
            dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))

        # realistically will not be able to generate that many frames of video... yet
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32)

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size,
                                   init_kernel_size), padding=(0, init_padding, init_padding))

        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # image conditioning
        if cond_dim is not None:
            SEQ_LENGTH = 257
            CLIP_VISION_SIZE = 1024

            if not self.medclip:
                self.fc_cond = nn.Linear(SEQ_LENGTH*CLIP_VISION_SIZE, CLIP_VISION_SIZE)

        # text conditioning
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim


        self.null_cond_emb = nn.Parameter(
            torch.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond=None,
        null_cond_prob=0.,
        focus_present_mask=None,
        # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        prob_focus_present=0.
    ):

        assert not (self.has_cond and not exists(cond)
                    ), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        x = self.init_conv(x)
        r = x.clone()

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        if not self.medclip:
            cond = self.fc_cond(cond.view(batch,-1)) if exists(self.fc_cond) else cond

        # classifier free guidance
        if self.has_cond:
            batch, device = x.shape[0], x.device

            mask = prob_mask_like((batch,), null_cond_prob, device=device)

            cond = torch.where(rearrange(mask, 'b -> b 1'),
                               self.null_cond_emb, cond)
            if self.cfg:
                cond = cond.long().cuda()
            t = torch.cat((t, cond), dim=-1)

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)

# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

def enforce_zero_terminal_snr(betas):
    """
    Corrects noise in diffusion schedulers.
    From: Common Diffusion Noise Schedules and Sample Steps are Flawed
    https://arxiv.org/pdf/2305.08891.pdf
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
        alphas_bar_sqrt_0 - alphas_bar_sqrt_T
    )

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        channels,
        timesteps,
        loss_type,
        l1_weight,
        perceptual_weight,
        discriminator_weight,
        name_dataset,
        text_use_bert_cls=False,
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.9,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.perceptual_weight = perceptual_weight
        self.name_dataset = name_dataset
        self.l1_weight = l1_weight
        self.discriminator_weight = discriminator_weight

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32
        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters
        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling
        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

        # discriminator for 2D X-ray images (adapted from 3D to 2D)
        self.netD = PatchDiscriminator(
            spatial_dims=2,      # Changed from 3 to 2 for 2D X-rays
            num_layers_d=4,
            num_channels=32,
            in_channels=1,       # Changed from 4 to 1 for grayscale X-ray
            out_channels=1,
            kernel_size=3
        )
        self.netD.cuda()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.optimizerD = Adam(params=self.netD.parameters(), lr=1e-4)

        # perceptual loss for 2D images
        self.perceptual_model = PerceptualLoss(spatial_dims=2, network_type="radimagenet_resnet50")
        self.perceptual_model.cuda()


    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond_ct=None, cond_scale=1.):
        # For 2D X-ray diffusion with CT conditioning, pass cond_ct directly to denoise_fn
        noise_pred = self.denoise_fn(x, t, cond_ct=cond_ct)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond_ct=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond_ct=cond_ct, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond_ct=None, cond_scale=1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long), cond_ct=cond_ct, cond_scale=cond_scale)

        return img

    @torch.inference_mode()
    def sample(self, cond_ct=None, cond_scale=1., batch_size=16):
        """
        Sample 2D X-ray images conditioned on CT.

        Args:
            cond_ct: VAE-encoded CT tensor of shape [B, 4, 128, 56, 56]
            cond_scale: conditioning scale (not used in current implementation)
            batch_size: batch size for sampling

        Returns:
            Generated X-ray images of shape [B, 1, 224, 224]
        """
        device = next(self.denoise_fn.parameters()).device

        # CT is already VAE-encoded, just move to device
        if cond_ct is not None:
            cond_ct = cond_ct.to(device)
            batch_size = cond_ct.shape[0]

        image_size = self.image_size
        channels = self.channels

        # Sample 2D X-ray: shape is (batch_size, channels, height, width)
        _sample = self.p_sample_loop(
            (batch_size, channels, image_size, image_size), cond_ct=cond_ct, cond_scale=cond_scale)

        # Unnormalize the output (from [-1, 1] to [0, 1])
        _sample = unnormalize_img(_sample)

        return _sample

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):

        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )
    def lpips_loss_fn(self, x_start, x_recon):
        """
        Perceptual loss for 2D X-ray images.

        Args:
            x_start: Original X-ray images [B, C, H, W]
            x_recon: Reconstructed X-ray images [B, C, H, W]

        Returns:
            Perceptual loss value
        """
        # Unnormalize from [-1, 1] to [0, 1] for perceptual loss
        x_start_unnorm = (x_start + 1.0) / 2.0
        x_recon_unnorm = (x_recon + 1.0) / 2.0

        # Convert grayscale to 3-channel for perceptual model if needed
        if x_start_unnorm.shape[1] == 1:
            x_start_unnorm = x_start_unnorm.repeat(1, 3, 1, 1)
            x_recon_unnorm = x_recon_unnorm.repeat(1, 3, 1, 1)

        # perceptual loss
        lpips_loss = self.perceptual_model(
            x_recon_unnorm.float(), x_start_unnorm.float()).mean() * self.perceptual_weight
        return lpips_loss

    def disc_loss_fn(self, x_real,x_fake):

        self.optimizerD.zero_grad(set_to_none=True)
        logits_fake = self.netD(x_fake.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.netD(x_real.contiguous().detach())[-1]
        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = self.discriminator_weight * discriminator_loss

        loss_d.backward(retain_graph=True)
        self.optimizerD.step()

        return loss_d


    def p_losses(self, x_start, t, cond_ct=None, noise=None, **kwargs):
        """
        Compute diffusion loss for 2D X-ray generation conditioned on CT.

        Args:
            x_start: Target X-ray images [B, C, H, W]
            t: Timestep indices [B]
            cond_ct: VAE-encoded CT condition [B, 4, 128, 56, 56]
            noise: Optional noise tensor

        Returns:
            Diffusion loss
        """
        b, c, h, w = x_start.shape
        device = x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Add noise to X-ray
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict noise using UNet with CT conditioning
        pred_noise = self.denoise_fn(x_noisy, t, cond_ct=cond_ct, **kwargs)
        x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=pred_noise)

        # Perceptual loss
        lpips_loss = 0
        if self.perceptual_weight > 0:
            lpips_loss = self.lpips_loss_fn(x_start, x_recon)

        # Discriminator loss
        disc_loss = 0
        if self.discriminator_weight > 0:
            disc_loss = self.disc_loss_fn(x_start,x_recon)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, pred_noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, pred_noise)
        elif self.loss_type == 'l1_lpips':
            loss = F.l1_loss(noise, pred_noise)*self.l1_weight + lpips_loss
        elif self.loss_type == 'l1_disc':
            loss = F.l1_loss(noise, pred_noise)*self.l1_weight + disc_loss
        elif self.loss_type == 'l1_lpips_disc':
            loss = F.l1_loss(noise, pred_noise)*self.l1_weight + lpips_loss + disc_loss
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        """
        Forward pass for training: diffuse X-ray conditioned on CT.

        Args:
            x: Dictionary containing 'ct' and 'cxr' tensors
                - ct: VAE-encoded CT [B, 4, 128, 56, 56]
                - cxr: X-ray images [B, 1, 224, 224]

        Returns:
            Diffusion loss
        """
        ct = x['ct'].cuda()      # CT is the condition [B, 4, 128, 56, 56]
        xray = x['cxr'].cuda()   # X-ray is the diffusion target [B, 1, 224, 224], already in [-1, 1]

        # X-ray is already normalized to [-1, 1] by preprocess_xray.py (per-image min-max)
        # No additional normalization needed here

        b, device, img_size = xray.shape[0], xray.device, self.image_size

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Apply diffusion on X-ray with CT as condition
        return self.p_losses(xray, t, cond_ct=ct, *args, **kwargs)

# trainer class
CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif
def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images

# gif -> (channels, frame, height, width) tensor
def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels=3,
        num_frames=16,
        horizontal_flip=False,
        force_num_frames=True,
        exts=['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(
            f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(
            cast_num_frames, frames=num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform=self.transform)
        return self.cast_num_frames_fn(tensor)

# trainer class


class Trainer(object):
    """
    Trainer class for CT-to-Xray diffusion model.

    Features:
    - Epoch-aware training loop with per-epoch wandb logging
    - Full validation on entire validation set every epoch
    - tqdm progress bars for training monitoring
    - wandb integration for experiment tracking
    - Comprehensive validation metrics (PSNR, SSIM, LPIPS, MAE, MSE)
    - Sample visualization with real vs generated comparisons
    - Final test evaluation after training completes
    """

    def __init__(
        self,
        diffusion_model,
        cfg,
        folder=None,
        dataset=None,
        val_dataset=None,
        test_dataset=None,
        *,
        ema_decay,
        train_batch_size,
        train_lr,
        train_num_steps,
        gradient_accumulate_every,
        amp,
        step_start_ema,
        update_ema_every,
        save_and_sample_every,
        results_folder,
        num_sample_rows,
        max_grad_norm,
        num_workers,
        lora,
        lora_first,
        use_wandb,
        wandb_project,
        wandb_run_name,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.lora = lora
        self.lora_first = lora_first

        self.cfg = cfg

        # Setup training dataset
        if dataset:
            self.ds = dataset
        else:
            assert folder is not None, 'Provide a folder path to the dataset'
            self.ds = Dataset(folder, diffusion_model.image_size,
                              channels=diffusion_model.channels,
                              num_frames=diffusion_model.num_frames)

        self.train_dl = DataLoader(self.ds, batch_size=train_batch_size,
                        shuffle=True, pin_memory=True,
                        num_workers=num_workers, prefetch_factor=2)
        self.steps_per_epoch = len(self.train_dl)

        # Setup validation dataset (non-cycled — iterate fully each epoch)
        self.val_ds = val_dataset
        self.val_dl = None
        if val_dataset:
            self.val_dl = DataLoader(self.val_ds, batch_size=train_batch_size,
                        shuffle=False, pin_memory=True,
                        num_workers=num_workers, prefetch_factor=2)

        # Setup test dataset (non-cycled — iterate fully once)
        self.test_ds = test_dataset
        self.test_dl = None
        if test_dataset:
            self.test_dl = DataLoader(self.test_ds, batch_size=train_batch_size,
                        shuffle=False, pin_memory=True,
                        num_workers=num_workers, prefetch_factor=2)

        print(f'Training dataset size: {len(self.ds)}')
        print(f'Steps per epoch: {self.steps_per_epoch}')
        if val_dataset:
            print(f'Validation dataset size: {len(self.val_ds)}')
        if test_dataset:
            print(f'Test dataset size: {len(self.test_ds)}')

        assert len(self.ds) > 0, 'Need at least 1 sample to start training'

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.train_lr = train_lr
        self.step = 0
        self.epoch = 0
        self.amp = amp
        self.scaler = GradScaler('cuda', enabled=amp)
        self.max_grad_norm = max_grad_norm
        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        # Initialize LPIPS metric for validation
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()
        self.lpips_metric.eval()

        # Wandb setup
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    'train_batch_size': train_batch_size,
                    'train_lr': train_lr,
                    'train_num_steps': train_num_steps,
                    'gradient_accumulate_every': gradient_accumulate_every,
                    'ema_decay': ema_decay,
                    'amp': amp,
                    'step_start_ema': step_start_ema,
                    'update_ema_every': update_ema_every,
                    'save_and_sample_every': save_and_sample_every,
                    'max_grad_norm': max_grad_norm,
                    'image_size': self.image_size,
                    'loss_type': diffusion_model.loss_type,
                    'timesteps': diffusion_model.num_timesteps,
                    'l1_weight': diffusion_model.l1_weight,
                    'perceptual_weight': diffusion_model.perceptual_weight,
                    'discriminator_weight': diffusion_model.discriminator_weight,
                    'dataset_size': len(self.ds),
                    'val_dataset_size': len(self.val_ds) if self.val_ds else 0,
                    'test_dataset_size': len(self.test_ds) if self.test_ds else 0,
                    'steps_per_epoch': self.steps_per_epoch,
                    'lora': lora,
                },
                resume='allow'
            )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset EMA model parameters to match the current model."""
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        """Update EMA model parameters."""
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        """Save model checkpoint."""
        data = {
            'step': self.step,
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        save_path = str(self.results_folder / f'model-{milestone}.pt')
        torch.save(data, save_path)
        tqdm.write(f'Saved checkpoint to {save_path}')

    def load(self, milestone, map_location=None, **kwargs):
        """Load model checkpoint."""
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'Need at least one milestone to load from'
            milestone = max(all_milestones)

        data = torch.load(milestone, map_location=map_location) if map_location else torch.load(milestone)

        self.step = data['step']
        self.epoch = data.get('epoch', self.step // self.steps_per_epoch)

        if self.lora:
            if self.lora_first:
                self.model.load_state_dict(data['model'], strict=False, **kwargs)
                self.ema_model.load_state_dict(data['ema'], strict=False, **kwargs)
                self.model.eval()
                self.ema_model.eval()
                inject_trainable_lora(self.model.denoise_fn)
                inject_trainable_lora(self.ema_model.denoise_fn)
            else:
                self.model.eval()
                self.ema_model.eval()
                inject_trainable_lora(self.model.denoise_fn)
                inject_trainable_lora(self.ema_model.denoise_fn)
                self.model.load_state_dict(data['model'], strict=False, **kwargs)
                self.ema_model.load_state_dict(data['ema'], strict=False, **kwargs)
        else:
            self.model.load_state_dict(data['model'], strict=False, **kwargs)
            self.ema_model.load_state_dict(data['ema'], strict=False, **kwargs)

        self.scaler.load_state_dict(data['scaler'])
        if 'optimizer' in data:
            self.opt.load_state_dict(data['optimizer'])

        print(f'Loaded checkpoint from step {self.step}')

    def compute_metrics(self, real_images, generated_images):
        """
        Compute validation metrics between real and generated images.

        Args:
            real_images: Ground truth X-ray images [B, 1, H, W] in range [0, 1]
            generated_images: Generated X-ray images [B, 1, H, W] in range [0, 1]

        Returns:
            Dictionary with computed metrics
        """
        metrics = {}
        batch_size = real_images.shape[0]

        # Ensure images are in [0, 1] range
        real_images = real_images.clamp(0, 1)
        generated_images = generated_images.clamp(0, 1)

        # PSNR, SSIM, MAE (computed per image, then averaged)
        psnr_values = []
        ssim_values = []
        mae_values = []

        for i in range(batch_size):
            real_np = real_images[i, 0].cpu().numpy()
            gen_np = generated_images[i, 0].cpu().numpy()

            # PSNR
            psnr_val = psnr_metric(real_np, gen_np, data_range=1.0)
            psnr_values.append(psnr_val)

            # SSIM
            ssim_val = ssim_metric(real_np, gen_np, data_range=1.0)
            ssim_values.append(ssim_val)

            # MAE (Mean Absolute Error)
            mae_val = np.abs(real_np - gen_np).mean()
            mae_values.append(mae_val)

        metrics['psnr'] = np.mean(psnr_values)
        metrics['ssim'] = np.mean(ssim_values)
        metrics['mae'] = np.mean(mae_values)

        # LPIPS (needs 3 channels and range [-1, 1])
        real_3ch = real_images.repeat(1, 3, 1, 1) * 2 - 1
        gen_3ch = generated_images.repeat(1, 3, 1, 1) * 2 - 1
        with torch.no_grad():
            lpips_val = self.lpips_metric(real_3ch.cuda(), gen_3ch.cuda())
        metrics['lpips'] = lpips_val.item()

        # MSE
        mse_val = F.mse_loss(real_images, generated_images)
        metrics['mse'] = mse_val.item()

        return metrics

    @torch.no_grad()
    def validate(self):
        """
        Run validation on the entire validation dataset.

        Returns:
            Dictionary with averaged validation metrics
        """
        self.ema_model.eval()

        all_metrics = {
            'val/psnr': [], 'val/ssim': [], 'val/lpips': [],
            'val/mae': [], 'val/mse': []
        }

        val_pbar = tqdm(self.val_dl, desc='Validation', unit='batch', leave=False)
        for val_data in val_pbar:
            ct_cond = val_data['ct'].cuda()
            real_xray = val_data['cxr'].cuda()

            # Convert real X-ray from [-1, 1] to [0, 1] to match sample() output
            real_xray = (real_xray + 1.0) / 2.0

            # Generate X-rays from CT condition (output is [0, 1])
            generated_xray = self.ema_model.sample(cond_ct=ct_cond, batch_size=ct_cond.shape[0])

            # Compute metrics (both in [0, 1])
            metrics = self.compute_metrics(real_xray, generated_xray)

            all_metrics['val/psnr'].append(metrics['psnr'])
            all_metrics['val/ssim'].append(metrics['ssim'])
            all_metrics['val/lpips'].append(metrics['lpips'])
            all_metrics['val/mae'].append(metrics['mae'])
            all_metrics['val/mse'].append(metrics['mse'])

            val_pbar.set_postfix({
                'psnr': f'{metrics["psnr"]:.2f}',
                'ssim': f'{metrics["ssim"]:.4f}'
            })

        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        return avg_metrics

    @torch.no_grad()
    def test(self):
        """
        Run final evaluation on the test set.

        This method should only be called once after training is complete.
        It evaluates the model on the entire test dataset and logs results.

        Returns:
            Dictionary with averaged test metrics
        """
        if self.test_ds is None:
            print("No test dataset available. Skipping test evaluation.")
            return None

        self.ema_model.eval()

        all_metrics = {
            'test/psnr': [], 'test/ssim': [], 'test/lpips': [],
            'test/mae': [], 'test/mse': []
        }

        print(f'\n{"="*60}')
        print('Running final evaluation on test set...')
        print(f'Test dataset size: {len(self.test_ds)}')
        print(f'{"="*60}\n')

        # Iterate over entire test dataset
        test_pbar = tqdm(self.test_dl, desc='Testing', unit='batch')

        for test_data in test_pbar:
            ct_cond = test_data['ct'].cuda()
            real_xray = test_data['cxr'].cuda()

            # Convert real X-ray from [-1, 1] to [0, 1] to match sample() output
            real_xray = (real_xray + 1.0) / 2.0

            # Generate X-rays from CT condition (output is [0, 1])
            generated_xray = self.ema_model.sample(cond_ct=ct_cond, batch_size=ct_cond.shape[0])

            # Compute metrics (both in [0, 1])
            metrics = self.compute_metrics(real_xray, generated_xray)

            all_metrics['test/psnr'].append(metrics['psnr'])
            all_metrics['test/ssim'].append(metrics['ssim'])
            all_metrics['test/lpips'].append(metrics['lpips'])
            all_metrics['test/mae'].append(metrics['mae'])
            all_metrics['test/mse'].append(metrics['mse'])

            # Update progress bar
            test_pbar.set_postfix({
                'psnr': f'{metrics["psnr"]:.2f}',
                'ssim': f'{metrics["ssim"]:.4f}'
            })

        # Average metrics over entire test set
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

        print(f'\n{"="*60}')
        print('Test Results:')
        print(f'  PSNR:  {avg_metrics["test/psnr"]:.2f}')
        print(f'  SSIM:  {avg_metrics["test/ssim"]:.4f}')
        print(f'  LPIPS: {avg_metrics["test/lpips"]:.4f}')
        print(f'  MAE:   {avg_metrics["test/mae"]:.4f}')
        print(f'  MSE:   {avg_metrics["test/mse"]:.4f}')
        print(f'{"="*60}\n')

        # Log to wandb
        if self.use_wandb:
            wandb.log(avg_metrics, step=self.step)

            # Create summary table
            wandb.run.summary.update({
                'final_test_psnr': avg_metrics['test/psnr'],
                'final_test_ssim': avg_metrics['test/ssim'],
                'final_test_lpips': avg_metrics['test/lpips'],
                'final_test_mae': avg_metrics['test/mae'],
                'final_test_mse': avg_metrics['test/mse'],
            })

        return avg_metrics

    @torch.no_grad()
    def save_and_log_samples(self, milestone):
        """Save checkpoint and generate samples for visualization."""
        self.ema_model.eval()

        # Get first batch from validation data for sampling
        val_data = next(iter(self.val_dl))
        ct_cond = val_data['ct'].cuda()
        real_xray = val_data['cxr'].cuda()  # raw [-1, 1]

        num_samples = min(self.num_sample_rows ** 2, ct_cond.shape[0], 4)
        ct_cond = ct_cond[:num_samples]
        real_xray = real_xray[:num_samples]

        # Generate samples (output is [0, 1])
        generated_xray = self.ema_model.sample(cond_ct=ct_cond, batch_size=num_samples)

        # Convert real X-ray from [-1, 1] to [0, 1] for metrics and display
        real_xray_01 = (real_xray + 1.0) / 2.0

        # Compute metrics (both in [0, 1])
        metrics = self.compute_metrics(real_xray_01, generated_xray)

        tqdm.write(
            f"[Milestone {milestone}] PSNR: {metrics['psnr']:.2f} | "
            f"SSIM: {metrics['ssim']:.4f} | LPIPS: {metrics['lpips']:.4f} | "
            f"MAE: {metrics['mae']:.4f}"
        )

        # Create comparison figure
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Real X-ray
            axes[i, 0].imshow(real_xray_01[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title('Real X-ray', fontsize=10)
            axes[i, 0].axis('off')

            # Generated X-ray
            axes[i, 1].imshow(generated_xray[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('Generated X-ray', fontsize=10)
            axes[i, 1].axis('off')

            # Difference map
            diff = torch.abs(real_xray_01[i, 0] - generated_xray[i, 0]).cpu().numpy()
            axes[i, 2].imshow(diff, cmap='hot')
            axes[i, 2].set_title('Absolute Difference', fontsize=10)
            axes[i, 2].axis('off')

        plt.suptitle(f'Step {self.step} | PSNR: {metrics["psnr"]:.2f} | SSIM: {metrics["ssim"]:.4f}', fontsize=12)
        plt.tight_layout()
        sample_path = str(self.results_folder / f'sample-{milestone}.png')
        plt.savefig(sample_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'samples/comparison': wandb.Image(sample_path),
                'samples/psnr': metrics['psnr'],
                'samples/ssim': metrics['ssim'],
                'samples/lpips': metrics['lpips'],
                'samples/mae': metrics['mae'],
                'samples/mse': metrics['mse'],
            }, step=self.step)

        # Save checkpoint
        self.save(milestone)

    def train(self):
        """
        Main training loop — epoch-based with per-epoch logging and validation.

        - Logs average training loss to wandb every epoch
        - Runs full validation on entire validation set every epoch
        - Saves checkpoints and samples at step intervals (save_and_sample_every)
        """
        total_epochs = self.train_num_steps // self.steps_per_epoch
        start_epoch = self.step // self.steps_per_epoch

        print(f'\n{"="*60}')
        print(f'Starting training from step {self.step} (epoch {start_epoch})')
        print(f'Total steps: {self.train_num_steps}')
        print(f'Steps per epoch: {self.steps_per_epoch}')
        print(f'Estimated total epochs: {total_epochs}')
        print(f'Batch size: {self.batch_size}')
        print(f'Gradient accumulation: {self.gradient_accumulate_every}')
        print(f'Effective batch size: {self.batch_size * self.gradient_accumulate_every}')
        print(f'Results folder: {self.results_folder}')
        print(f'Wandb logging: {self.use_wandb}')
        print(f'{"="*60}\n')

        # Create progress bar for total training
        pbar = tqdm(
            initial=self.step,
            total=self.train_num_steps,
            desc='Training',
            unit='step',
            dynamic_ncols=True
        )

        while self.step < self.train_num_steps:
            # ---- Epoch start ----
            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            epoch_grad_norm = 0.0

            for batch_data in self.train_dl:
                if self.step >= self.train_num_steps:
                    break

                accumulated_loss = 0.0

                # Gradient accumulation loop
                for _ in range(self.gradient_accumulate_every):
                    with autocast('cuda', enabled=self.amp):
                        loss = self.model(batch_data)
                        scaled_loss = loss / self.gradient_accumulate_every

                    self.scaler.scale(scaled_loss).backward()
                    accumulated_loss += loss.item()

                # Average loss for this step
                step_loss = accumulated_loss / self.gradient_accumulate_every
                epoch_loss += step_loss
                epoch_steps += 1

                # Gradient clipping
                if exists(self.max_grad_norm):
                    self.scaler.unscale_(self.opt)
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    epoch_grad_norm += grad_norm.item()
                else:
                    grad_norm = None

                # Optimizer step
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                # EMA update
                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                # Update progress bar
                pbar.set_postfix({
                    'epoch': self.epoch,
                    'loss': f'{step_loss:.4f}',
                    'lr': f'{self.opt.param_groups[0]["lr"]:.2e}'
                })
                pbar.update(1)

                # Save and sample (step-based)
                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every
                    self.save_and_log_samples(milestone)
                    self.model.train()

                self.step += 1

            # ---- Epoch end ----
            if epoch_steps == 0: # empty dataloader
                break

            avg_epoch_loss = epoch_loss / epoch_steps
            avg_epoch_grad_norm = epoch_grad_norm / epoch_steps if exists(self.max_grad_norm) else None

            # Log training metrics for this epoch
            epoch_log = {
                'train/epoch_loss': avg_epoch_loss,
                'train/epoch': self.epoch,
                'train/step': self.step,
                'train/lr': self.opt.param_groups[0]['lr'],
            }
            if avg_epoch_grad_norm is not None:
                epoch_log['train/grad_norm'] = avg_epoch_grad_norm

            tqdm.write(
                f"[Epoch {self.epoch}] Train Loss: {avg_epoch_loss:.4f} | "
                f"Step: {self.step}"
            )

            # Run full validation every epoch
            if self.val_dl is not None:
                val_metrics = self.validate()

                tqdm.write(
                    f"[Epoch {self.epoch}] Val PSNR: {val_metrics['val/psnr']:.2f} | "
                    f"Val SSIM: {val_metrics['val/ssim']:.4f} | "
                    f"Val LPIPS: {val_metrics['val/lpips']:.4f} | "
                    f"Val MAE: {val_metrics['val/mae']:.4f}"
                )
                epoch_log.update(val_metrics)

            if self.use_wandb:
                wandb.log(epoch_log, step=self.step)

            self.epoch += 1

        pbar.close()

        # Final save
        self.save('final')
        tqdm.write('\nTraining completed!')

        # Run final evaluation on test set
        self.test()

        if self.use_wandb:
            wandb.finish()
