"""
UNetV2: Enhanced 2D UNet for X-ray diffusion with CT conditioning.

Improvements over the original UNet (unet.py):
  1. ResNet blocks (like the Unet3D in the original paper) instead of MONAI UnetBasicBlock.
     Each block: Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU + residual skip.
     FiLM conditioning (scale-shift) from time embedding is applied after the first GroupNorm.
  2. Spatial linear attention at every encoder/decoder level (like the original paper).
     O(N) linear attention instead of O(N^2) -- efficient even at high resolutions.
  3. FiLM conditioning for CT features (scale-shift) instead of additive injection.
     CT features produce per-spatial-location scale and shift, applied via: x = x * (1 + scale) + shift.
  4. Same CT3Dto2DProjector and same forward interface: forward(x_t, t, cond_ct).

CT conditioning is injected only where spatial sizes match (56x56, 28x28, 14x14).
Levels at 224x224 and 112x112 have no CT conditioning (CT projector outputs at 56x56 max).

Architecture (default hid_chs=[32, 64, 128, 256, 512]):
  Input: [B, 1, 224, 224]

  InConv: Conv2d(1->32, 7x7) -> [B, 32, 224x224], save residual r

  Enc0: ResBlock(32->64) + ResBlock(64->64) + LinearAttn + save skip -> Downsample -> [B, 64, 112x112]
  Enc1: ResBlock(64->128) + ResBlock(128->128) + LinearAttn + save skip -> Downsample -> CT FiLM(56x56) -> [B, 128, 56x56]
  Enc2: ResBlock(128->256) + ResBlock(256->256) + LinearAttn + save skip -> Downsample -> CT FiLM(28x28) -> [B, 256, 28x28]
  Enc3: ResBlock(256->512) + ResBlock(512->512) + LinearAttn + save skip -> Downsample -> CT FiLM(14x14) -> [B, 512, 14x14]

  Bottleneck: ResBlock(512->512) -> CT FiLM(14x14) -> FullAttention -> ResBlock(512->512) -> [B, 512, 14x14]

  Dec0: Upsample(14->28) -> cat(skip3) [1024ch] -> ResBlock(1024->256) -> ResBlock(256->256) -> CT FiLM(28x28) -> LinearAttn -> [B, 256, 28x28]
  Dec1: Upsample(28->56) -> cat(skip2) [512ch] -> ResBlock(512->128) -> ResBlock(128->128) -> CT FiLM(56x56) -> LinearAttn -> [B, 128, 56x56]
  Dec2: Upsample(56->112) -> cat(skip1) [256ch] -> ResBlock(256->64) -> ResBlock(64->64) -> LinearAttn -> [B, 64, 112x112]
  Dec3: Upsample(112->224) -> cat(skip0) [128ch] -> ResBlock(128->32) -> ResBlock(32->32) -> LinearAttn -> [B, 32, 224x224]

  OutConv: cat(x, r) [64ch] -> ResBlock(64->32) -> Conv2d(32->1) -> [B, 1, 224x224]
"""
from ddpm.time_embedding import TimeEmbbeding
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# ============================================================
# Building Blocks (adapted from Unet3D in diffusion.py -> 2D)
# ============================================================
class Block2D(nn.Module):
    """Single conv block: Conv2d -> GroupNorm -> SiLU, with optional FiLM scale-shift."""
    def __init__(self, dim_in, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)
class ResnetBlock2D(nn.Module):
    """
    ResNet block with FiLM time conditioning (like the original paper's ResnetBlock).
    Structure:
      Block1(in -> out, FiLM from time_emb) -> Block2(out -> out) + residual(in -> out)
    The time embedding is projected to 2*out_dim and split into (scale, shift)
    which modulate the output of Block1 via: x = x * (1 + scale) + shift.
    """
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        # Time FiLM: project time embedding to scale+shift for dim_out channels
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None
        self.block1 = Block2D(dim_in, dim_out, groups=groups)
        self.block2 = Block2D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
class SpatialLinearAttention2D(nn.Module):
    """
    Linear attention over spatial dimensions (like the original paper's SpatialLinearAttention).
    Uses softmax-normalized Q and K for O(N) complexity via the kernel trick.
    Input/output: [B, C, H, W].
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads),
            qkv
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)
class FullSpatialAttention2D(nn.Module):
    """
    Full quadratic spatial attention (used in bottleneck only, like the original paper).
    Input/output: [B, C, H, W].
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads),
            qkv
        )
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) c -> b (h c) x y', x=h, y=w)
        return self.to_out(out)
class LayerNorm2D(nn.Module):
    """Channel-wise layer norm for 2D feature maps."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma
class PreNorm2D(nn.Module):
    """Apply LayerNorm before a module."""
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm2D(dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
class Residual(nn.Module):
    """Wrap a module with a residual connection."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x
class Downsample2D(nn.Module):
    """Spatial downsample by 2x using strided convolution."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)
class Upsample2D(nn.Module):
    """Spatial upsample by 2x using transposed convolution."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)
# ============================================================
# CT 3D -> 2D Projector (same as unet.py)
# ============================================================
class CT3Dto2DProjector(nn.Module):
    def __init__(self, in_ch=4, out_ch=256, base_ch=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
        )
        self.depth_compress = nn.Sequential(
            nn.Conv3d(base_ch, base_ch * 2, kernel_size=(8, 1, 1), stride=(8, 1, 1)),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
            nn.Conv3d(base_ch * 2, out_ch, kernel_size=(16, 1, 1), stride=(16, 1, 1)),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
    def forward(self, ct):
        """ct VAE latent: [B, 4, 128, 56, 56] -> [B, out_ch, 56, 56]"""
        x = self.encoder(ct)
        x = self.depth_compress(x)   # [B, out_ch, 1, 56, 56]
        return x.squeeze(2)          # [B, out_ch, 56, 56]
# ============================================================
# CT FiLM Conditioning Module
# ============================================================
class CTFiLMConditioner(nn.Module):
    """
    Generates per-spatial-location FiLM parameters (scale, shift) from CT features.
    Takes CT feature map and produces scale & shift of the target channel dimension.
    Applied as: x = x * (1 + scale) + shift
    This is more expressive than additive conditioning (x = x + cond) because:
    - Scale can amplify or suppress features based on CT content
    - Shift can add CT-dependent bias
    - Together they provide affine modulation at each spatial location
    """
    def __init__(self, cond_ch, target_ch):
        super().__init__()
        self.film_proj = nn.Sequential(
            nn.Conv2d(cond_ch, target_ch * 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(target_ch * 2, target_ch * 2, kernel_size=1),
        )
    def forward(self, cond_feat, x):
        """
        Args:
            cond_feat: CT feature map [B, cond_ch, H, W]
            x: UNet feature map to modulate [B, target_ch, H, W]
        Returns:
            Modulated x: [B, target_ch, H, W]
        """
        film_params = self.film_proj(cond_feat)
        scale, shift = film_params.chunk(2, dim=1)
        return x * (1 + scale) + shift
# ============================================================
# UNetV2: Enhanced 2D UNet
# ============================================================
class UNetV2(nn.Module):
    """
    Enhanced 2D UNet for noise prediction in X-ray diffusion, conditioned on CT.
    Key differences from the original UNet (unet.py):
      - ResNet blocks with FiLM time conditioning (like the original paper)
      - Spatial linear attention at every encoder/decoder level
      - Full spatial attention in bottleneck
      - CT conditioning via FiLM (scale-shift) instead of additive
    Same forward interface: forward(x_t, t, cond_ct) -> noise prediction
    """
    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        spatial_dims=2,  # kept for interface compatibility (always 2)
        hid_chs=(32, 64, 128, 256, 512),
        groups=8,
        attn_heads=4,
        attn_dim_head=32,
        ct_projector_ch=256,
        time_embedder=TimeEmbbeding,
        time_embedder_kwargs=None,
    ):
        super().__init__()
        if time_embedder_kwargs is None:
            time_embedder_kwargs = {}
        hid_chs = list(hid_chs)
        self.hid_chs = hid_chs
        num_levels = len(hid_chs) - 1  # 4 encoder levels for 5 channel sizes
        # ---- Time embedding ----
        self.time_embedder = time_embedder(**time_embedder_kwargs)
        time_emb_dim = self.time_embedder.emb_dim
        # ---- Initial convolution: in_ch -> hid_chs[0] ----
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_ch, hid_chs[0], kernel_size=7, padding=3),
            nn.GroupNorm(groups, hid_chs[0]),
            nn.SiLU(),
        )
        # ---- Encoder ----
        # Each level: ResBlock1 + ResBlock2 + [CT FiLM] + Attention + Downsample
        self.enc_blocks1 = nn.ModuleList()
        self.enc_blocks2 = nn.ModuleList()
        self.enc_attns = nn.ModuleList()
        self.enc_downsamples = nn.ModuleList()
        for i in range(num_levels):
            ch_in = hid_chs[i]
            ch_out = hid_chs[i + 1]
            self.enc_blocks1.append(
                ResnetBlock2D(ch_in, ch_out, time_emb_dim=time_emb_dim, groups=groups))
            self.enc_blocks2.append(
                ResnetBlock2D(ch_out, ch_out, time_emb_dim=time_emb_dim, groups=groups))
            self.enc_attns.append(
                Residual(PreNorm2D(ch_out,
                    SpatialLinearAttention2D(ch_out, heads=attn_heads, dim_head=attn_dim_head))))
            self.enc_downsamples.append(Downsample2D(ch_out))
        # ---- Bottleneck ----
        mid_ch = hid_chs[-1]
        self.mid_block1 = ResnetBlock2D(mid_ch, mid_ch, time_emb_dim=time_emb_dim, groups=groups)
        self.mid_attn = Residual(PreNorm2D(mid_ch,
            FullSpatialAttention2D(mid_ch, heads=attn_heads, dim_head=attn_dim_head)))
        self.mid_block2 = ResnetBlock2D(mid_ch, mid_ch, time_emb_dim=time_emb_dim, groups=groups)
        # ---- Decoder ----
        # Each level flow: Upsample -> concat skip -> ResBlock1 -> ResBlock2 -> [CT FiLM] -> Attention
        self.dec_blocks1 = nn.ModuleList()
        self.dec_blocks2 = nn.ModuleList()
        self.dec_attns = nn.ModuleList()
        self.dec_upsamples = nn.ModuleList()

        for i in reversed(range(num_levels)):
            ch_skip = hid_chs[i + 1]  # from encoder skip (saved before downsample)
            ch_from_deeper = hid_chs[i + 1]  # channel dim entering this decoder level
            ch_out = hid_chs[i]
            # Upsample operates on features from deeper level (before concat)
            self.dec_upsamples.append(Upsample2D(ch_from_deeper))
            # After concat: ch_from_deeper + ch_skip channels
            self.dec_blocks1.append(
                ResnetBlock2D(ch_from_deeper + ch_skip, ch_out, time_emb_dim=time_emb_dim, groups=groups))
            self.dec_blocks2.append(
                ResnetBlock2D(ch_out, ch_out, time_emb_dim=time_emb_dim, groups=groups))
            self.dec_attns.append(
                Residual(PreNorm2D(ch_out,
                    SpatialLinearAttention2D(ch_out, heads=attn_heads, dim_head=attn_dim_head))))
        # ---- Final convolution ----
        # Concat with initial residual: hid_chs[0] + hid_chs[0]
        self.final_block = ResnetBlock2D(
            hid_chs[0] * 2, hid_chs[0], time_emb_dim=time_emb_dim, groups=groups)
        self.final_conv = nn.Conv2d(hid_chs[0], out_ch, kernel_size=1)
        # ---- CT conditioning via FiLM ----
        # CT projector: [B, 4, 128, 56, 56] -> [B, ct_projector_ch, 56, 56]
        self.ct_projector = CT3Dto2DProjector(in_ch=4, out_ch=ct_projector_ch)

        # Spatial sizes AFTER downsampling at each encoder level:
        #   after enc level 0 downsample: 112x112  -- no CT (CT is 56x56)
        #   after enc level 1 downsample: 56x56    -- CT at 56x56
        #   after enc level 2 downsample: 28x28    -- CT pooled to 28x28
        #   after enc level 3 downsample: 14x14    -- CT pooled to 14x14 (= bottleneck input)
        #
        # CT FiLM is applied AFTER downsample on the features entering the next level.
        # This ensures spatial sizes match.

        # Encoder FiLM conditioners (applied after downsample, keyed by level that produced them)
        # After level 1 downsample: 56x56 features with hid_chs[2]=128 channels
        # After level 2 downsample: 28x28 features with hid_chs[3]=256 channels
        # After level 3 downsample: 14x14 features with hid_chs[4]=512 channels (= bottleneck)
        self.enc_ct_film = nn.ModuleDict({
            '1': CTFiLMConditioner(ct_projector_ch, hid_chs[2]),   # 56x56, 128ch
            '2': CTFiLMConditioner(ct_projector_ch, hid_chs[3]),   # 28x28, 256ch
            '3': CTFiLMConditioner(ct_projector_ch, hid_chs[4]),   # 14x14, 512ch
        })

        # Bottleneck FiLM conditioner: 14x14, 512ch (applied after mid_block1)
        self.mid_ct_film = CTFiLMConditioner(ct_projector_ch, mid_ch)

        # Decoder FiLM conditioners (applied after ResBlocks, before attention)
        # Decoder flow: upsample -> concat skip -> ResBlocks -> CT FiLM -> Attention
        # Decoder goes deepest->shallowest (reversed encoder order):
        #   dec[0]: upsample 14->28, ResBlocks output hid_chs[3]=256ch at 28x28
        #   dec[1]: upsample 28->56, ResBlocks output hid_chs[2]=128ch at 56x56
        #   dec[2]: upsample 56->112, ResBlocks output hid_chs[1]=64ch at 112x112 -- no CT
        #   dec[3]: upsample 112->224, ResBlocks output hid_chs[0]=32ch at 224x224 -- no CT
        self.dec_ct_film = nn.ModuleDict({
            '0': CTFiLMConditioner(ct_projector_ch, hid_chs[3]),   # 28x28, 256ch
            '1': CTFiLMConditioner(ct_projector_ch, hid_chs[2]),   # 56x56, 128ch
        })
    def forward(self, x_t, t, cond_ct, self_cond=None, **kwargs):
        """
        Args:
            x_t: Noisy X-ray [B, 1, 224, 224]
            t: Timestep [B]
            cond_ct: VAE-encoded CT [B, 4, 128, 56, 56]
        Returns:
            Predicted noise [B, 1, 224, 224]
        """
        # ---- Time embedding ----
        time_emb = self.time_embedder(t)  # [B, emb_dim]
        # ---- CT conditioning: project 3D CT to 2D feature map ----
        ct_feat = self.ct_projector(cond_ct)  # [B, 256, 56, 56]
        # Build CT features at multiple spatial scales
        ct_56 = ct_feat                           # [B, 256, 56, 56]
        ct_28 = F.avg_pool2d(ct_feat, 2)          # [B, 256, 28, 28]
        ct_14 = F.avg_pool2d(ct_28, 2)            # [B, 256, 14, 14]
        # Map encoder level index -> CT feature (None where no CT conditioning)
        enc_ct = {1: ct_56, 2: ct_28, 3: ct_14}
        # Map decoder index -> CT feature (after upsample, before ResBlocks)
        # dec[0]: upsample 14->28, process at 28x28 -> ct_28
        # dec[1]: upsample 28->56, process at 56x56 -> ct_56
        # dec[2]: upsample 56->112 -> no CT
        # dec[3]: upsample 112->224 -> no CT
        dec_ct = {0: ct_28, 1: ct_56}
        # ---- Initial convolution ----
        x = self.init_conv(x_t)  # [B, 32, 224, 224]
        r = x.clone()            # Save for final residual connection
        # ---- Encoder ----
        skips = []
        for i in range(len(self.enc_blocks1)):
            x = self.enc_blocks1[i](x, time_emb)
            x = self.enc_blocks2[i](x, time_emb)
            x = self.enc_attns[i](x)
            skips.append(x)
            x = self.enc_downsamples[i](x)

            # CT FiLM conditioning AFTER downsample (spatial sizes now match CT)
            # After level 1 downsample: 56x56, after level 2: 28x28, after level 3: 14x14
            if i in enc_ct:
                x = self.enc_ct_film[str(i)](enc_ct[i], x)
        # ---- Bottleneck ----
        x = self.mid_block1(x, time_emb)
        x = self.mid_ct_film(ct_14, x)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)
        # ---- Decoder ----
        for i in range(len(self.dec_blocks1)):
            # Upsample first to match skip connection spatial size
            x = self.dec_upsamples[i](x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks1[i](x, time_emb)
            x = self.dec_blocks2[i](x, time_emb)
            # CT FiLM conditioning (levels 0, 1, 2 only)
            if i in dec_ct:
                x = self.dec_ct_film[str(i)](dec_ct[i], x)
            x = self.dec_attns[i](x)
        # ---- Final convolution ----
        x = torch.cat([x, r], dim=1)  # [B, 64, 224, 224]
        x = self.final_block(x, time_emb)
        return self.final_conv(x)
