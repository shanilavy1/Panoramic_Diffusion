from ddpm.time_embedding import TimeEmbbeding
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import (
    UnetBasicBlock,
    UnetUpBlock,
    UnetOutBlock,
)
from monai.networks.layers.utils import get_act_layer

# CT 3D → 2D Projector
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
        """
        ct VAE latent: [B, 4, 128, 32, 32]
        returns: [B, out_ch, 32, 32]
        """
        x = self.encoder(ct)
        x = self.depth_compress(x)   # [B, out_ch, 1, 32, 32], compress depth (128 → 1) and produce rich feature channels
        return x.squeeze(2)          # [B, 256, 32, 32]


# Down Block
class DownBlock(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_ch,
        out_ch,
        time_emb_dim,
        act_name=("swish", {}),
        **kwargs
    ):
        super(DownBlock, self).__init__()

        self.loca_time_embedder = nn.Sequential(
            get_act_layer(name=act_name),
            nn.Linear(time_emb_dim, in_ch*2) # in_ch for additive time embedding
        )

        self.down_op = UnetBasicBlock(
            spatial_dims,
            in_ch,
            out_ch,
            act_name=act_name,
            **kwargs
        )

    def forward(self, x, time_emb, cond_map=None):
        b, c, *_ = x.shape
        sp_dim = x.ndim - 2

        # ---- Time embedding ----
        time_emb = self.loca_time_embedder(time_emb)
        time_emb = time_emb.reshape(b, 2*c, *((1,) * sp_dim)) #c for additive time embedding

        #  x = x + time_emb is additive time embedding
        scale, shift = time_emb.chunk(2, dim=1)
        x = x * (scale + 1) + shift

        # ---- CT conditioning ----
        if cond_map is not None:
            x = x + cond_map

        # ----------- Image ---------
        y = self.down_op(x)
        return y


# Up Block
class UpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims,
        skip_ch,
        enc_ch,
        time_emb_dim,
        act_name=("swish", {}),
        **kwargs
    ):
        super(UpBlock, self).__init__()

        self.up_op = UnetUpBlock(
            spatial_dims,
            enc_ch,
            skip_ch,
            act_name=act_name,
            **kwargs
        )

        self.loca_time_embedder = nn.Sequential(
            get_act_layer(name=act_name),
            nn.Linear(time_emb_dim, enc_ch*2), # enc_ch for additive time embedding
        )

    def forward(self, x_skip, x_enc, time_emb, cond_map=None):
        b, c, *_ = x_enc.shape
        sp_dim = x_enc.ndim - 2

        # ---- Time embedding ----
        time_emb = self.loca_time_embedder(time_emb)
        time_emb = time_emb.reshape(b, 2*c, *((1,) * sp_dim)) #c for additive time embedding

        #x_enc = x_enc + time_emb is additive time embedding
        scale, shift = time_emb.chunk(2, dim=1)
        x_enc = x_enc * (1 + scale) + shift

        # ---- CT conditioning ----
        if cond_map is not None:
            x_enc = x_enc + cond_map

        # ----------- Image -------------
        y = self.up_op(x_enc, x_skip)

        return y


# UNet
class UNet(nn.Module):
    def __init__(
        self,
        in_ch=1, # the xray has 1 channels
        out_ch=1, # predict the noise
        spatial_dims=2,
        hid_chs=[32, 64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_sizes=None,
        act_name=("SWISH", {}),
        norm_name=("INSTANCE", {"affine": True}),
        time_embedder=TimeEmbbeding,
        time_embedder_kwargs={},
        deep_ver_supervision=False, # Deep supervision produce outputs at multiple decoder depths
        estimate_variance=False, # output double channels to estimate variance in addition to epsilon
        use_self_conditioning=False, # The model is fed its own previous prediction as extra input.
        **kwargs
    ):
        super().__init__()

        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = strides[1:]

        # ---- Time embedding ----
        self.time_embedder = time_embedder(**time_embedder_kwargs)

        # ----------- In-Convolution ------------
        in_ch = in_ch * 2 if use_self_conditioning else in_ch
        self.inc = UnetBasicBlock(
            spatial_dims,
            in_ch,
            hid_chs[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            act_name=act_name,
            norm_name=norm_name,
            **kwargs
        )

        # ---- Encoder ----
        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims,
                hid_chs[i - 1],
                hid_chs[i],
                time_emb_dim=self.time_embedder.emb_dim,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                act_name=act_name,
                norm_name=norm_name,
                **kwargs
            )
            for i in range(1, len(strides))
        ])

        # ---- Decoder ----
        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims,
                hid_chs[i],
                hid_chs[i + 1],
                time_emb_dim=self.time_embedder.emb_dim,
                kernel_size=kernel_sizes[i + 1],
                stride=strides[i + 1],
                act_name=act_name,
                norm_name=norm_name,
                upsample_kernel_size=upsample_kernel_sizes[i],
                **kwargs
            )
            for i in range(len(strides) - 1)
        ])

        # --------------- Out-Convolution ----------------
        out_ch_hor = out_ch * 2 if estimate_variance else out_ch
        self.outc = UnetOutBlock(
            spatial_dims, hid_chs[0], out_ch_hor, dropout=None)
        if isinstance(deep_ver_supervision, bool):
            deep_ver_supervision = len(
                strides) - 2 if deep_ver_supervision else 0
        self.outc_ver = nn.ModuleList([
            UnetOutBlock(spatial_dims, hid_chs[i], out_ch, dropout=None)
            for i in range(1, deep_ver_supervision + 1)
        ])

        # ---- CT conditioning ----
        self.ct_projector = CT3Dto2DProjector(in_ch=4) #ct_latent: [4, 128, 32, 32]

        self.cond_down = nn.ModuleList([
            nn.Conv2d(256, hid_chs[0], kernel_size=1),
            nn.Conv2d(256, hid_chs[1], kernel_size=1),
            nn.Conv2d(256, hid_chs[2], kernel_size=1),
            nn.Conv2d(256, hid_chs[3], kernel_size=1),
        ]) # no spatial condition at the bottleneck (512 channels)

    def forward(self, x_t, t, cond_ct, self_cond=None,**kwargs):
        # x_t [B, C, H, W]
        # t [B,]

        # ---- CT conditioning ----
        ct_feat = self.ct_projector(cond_ct)  # [B, 256, 32, 32]

        cond_feats = []
        c = ct_feat
        for i in range(len(self.cond_down)):
            cond_feats.append(self.cond_down[i](c))
            if i < len(self.cond_down) - 1:
                c = F.avg_pool2d(c, kernel_size=2)

        # -------- In-Convolution --------------
        x = [None for _ in range(len(self.encoders)+1)]
        x_t = torch.cat([x_t, self_cond], dim=1) if self_cond is not None else x_t
        x[0] = self.inc(x_t)

        # -------- Time Embedding (Gloabl) -----------
        time_emb = self.time_embedder(t) # [B, C]

        # --------- Encoder --------------
        for i in range(len(self.encoders)):
            x[i + 1] = self.encoders[i](x[i], time_emb, cond_map=cond_feats[i])

        # -------- Decoder -----------
        for i in range(len(self.decoders), 0, -1):
            x[i - 1] = self.decoders[i - 1](x[i - 1], x[i], time_emb, cond_map=cond_feats[i - 1])

        # ---------Out-Convolution ------------
        y_hor = self.outc(x[0])

        return y_hor
