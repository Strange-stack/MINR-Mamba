import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def index_reverse(index):
    """Build the inverse permutation for a per-batch index tensor."""
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1], device=index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    """Reorder sequence tokens with the provided index tensor."""
    dim = index.dim()
    assert x.shape[:dim] == index.shape, (
        f'x ({x.shape}) and index ({index.shape}) shape incompatible')

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)
    return torch.gather(x, dim=dim - 1, index=index)


class DWConv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                dilation=1,
                groups=hidden_features,
            ),
            nn.GELU(),
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(
            x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = DWConv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        return self.fc2(x)


class ASSM(nn.Module):
    """Attentive state-space module adapted from MambaIRv2."""

    def __init__(self, dim, d_state, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        hidden = int(self.dim * mlp_ratio)
        self.d_state = d_state
        self.selectiveScan = SelectiveScan(d_model=hidden, d_state=d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(nn.Conv2d(self.dim, hidden, 1, 1, 0))
        self.CPE = nn.Sequential(nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden))

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x, x_size, token):
        b, n, c = x.shape
        h, w = x_size

        full_embedding = self.embeddingB.weight @ token.weight
        pred_route = self.route(x)
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)

        prompt = torch.matmul(cls_policy, full_embedding).view(b, n, self.d_state)
        semantic_index = torch.argmax(cls_policy.detach(), dim=-1).view(b, n)
        _, sort_indices = torch.sort(semantic_index, dim=-1, stable=False)
        reverse_indices = index_reverse(sort_indices)

        x = x.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        x = x.view(b, x.shape[1], -1).contiguous().permute(0, 2, 1)

        x = semantic_neighbor(x, sort_indices)
        x = self.selectiveScan(x, prompt)
        x = self.out_proj(self.out_norm(x))
        return semantic_neighbor(x, reverse_indices)


class SelectiveScan(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2.,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
                      bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init,
                         dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f'Unsupported dt_init: {dt_init}')

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) *
            (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        a = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        a_log = torch.log(a)
        if copies > 1:
            a_log = repeat(a_log, "d n -> r d n", r=copies)
            if merge:
                a_log = a_log.flatten(0, 1)
        a_log = nn.Parameter(a_log)
        a_log._no_weight_decay = True
        return a_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        d = torch.ones(d_inner, device=device)
        if copies > 1:
            d = repeat(d, "n1 -> r n1", r=copies)
            if merge:
                d = d.flatten(0, 1)
        d = nn.Parameter(d)
        d._no_weight_decay = True
        return d

    def forward_core(self, x, prompt):
        b, length, channels = x.shape
        scan_count = 1
        xs = x.permute(0, 2, 1).view(b, scan_count, channels, length).contiguous()

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            xs.view(b, scan_count, -1, length),
            self.x_proj_weight,
        )
        dts, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts.view(b, scan_count, -1, length),
            self.dt_projs_weight,
        )

        xs = xs.float().view(b, -1, length)
        dts = dts.contiguous().float().view(b, -1, length)
        bs = bs.float().view(b, scan_count, -1, length)
        cs = cs.float().view(b, scan_count, -1, length) + prompt
        ds = self.Ds.float().view(-1)
        a_s = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs,
            dts,
            a_s,
            bs,
            cs,
            ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(b, scan_count, -1, length)
        assert out_y.dtype == torch.float
        return out_y[:, 0]

    def forward(self, x, prompt, **kwargs):
        b, length, channels = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, channels, length)
        y = self.forward_core(x, prompt)
        return y.permute(0, 2, 1).contiguous()


class LFSSBlock(nn.Module):
    """Lightweight feature state-space block used by MINR-Mamba."""

    def __init__(self, dim, d_state, inner_rank, num_tokens,
                 convffn_kernel_size, mlp_ratio, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.inner_rank = inner_rank
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.scale2 = nn.Parameter(1e-4 * torch.ones(dim), requires_grad=True)

        self.assm = ASSM(
            self.dim,
            d_state,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio,
        )
        self.convffn2 = ConvFFN(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            kernel_size=convffn_kernel_size,
        )
        self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)

    def forward(self, x):
        b, c, h, w = x.shape
        x_size = [h, w]
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()

        shortcut = x
        x_assm = self.assm(self.norm3(x), x_size, self.embeddingA) + x
        x = x_assm + self.convffn2(self.norm4(x_assm), x_size)
        x = shortcut * self.scale2 + x
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()


def rgb_to_hsv_batch(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc, _ = rgb.max(dim=1)
    minc, _ = rgb.min(dim=1)
    v = maxc
    deltac = maxc - minc + 1e-8

    s = deltac / (maxc + 1e-8)
    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac

    h = torch.zeros_like(v)
    h = torch.where(maxc == r, (bc - gc) % 6, h)
    h = torch.where(maxc == g, 2.0 + rc - bc, h)
    h = torch.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    return torch.stack([h, s, v], dim=1)


def hsv_to_rgb_batch(hsv):
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    h = h * 6.0
    i = torch.floor(h).long() % 6
    f = h - torch.floor(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    values = (
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    )
    for idx, (rv, gv, bv) in enumerate(values):
        mask = i == idx
        r[mask], g[mask], b[mask] = rv[mask], gv[mask], bv[mask]
    return torch.stack([r, g, b], dim=1)


INR_HIDDEN_DIMS = [256, 256, 256]
POSITION_ENCODING_LEVELS = 4


def make_coord(shape, ranges=None, flatten=True, device=None):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n, device=device).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        last_dim = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            last_dim = hidden
        layers.append(nn.Linear(last_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class INR(nn.Module):
    """Coordinate MLP branch used for single-channel V reconstruction."""

    def __init__(self, dim, out_dim=1, local_ensemble=True, feat_unfold=True,
                 cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        imnet_in_dim = dim * 9 if feat_unfold else dim
        imnet_in_dim += 2 + 4 * POSITION_ENCODING_LEVELS
        if cell_decode:
            imnet_in_dim += 2
        self.imnet = MLP(imnet_in_dim, out_dim, INR_HIDDEN_DIMS)

    def query(self, inp, coord, cell=None):
        feat = inp
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst, vy_lst, eps_shift = [-1, 1], [-1, 1], 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 1 / feat.shape[-2]
        ry = 1 / feat.shape[-1]
        feat_coord = make_coord(
            feat.shape[-2:], flatten=False, device=feat.device
        ).permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                bs, channels, height, width = feat.shape
                q_feat = feat.view(bs, channels, -1).permute(0, 2, 1)
                q_coord = feat_coord.view(bs, 2, -1).permute(0, 2, 1)
                points_enc = self.positional_encoding(
                    q_coord, levels=POSITION_ENCODING_LEVELS)
                q_coord = torch.cat([q_coord, points_enc], dim=-1)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                mlp_input = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    mlp_input = torch.cat([mlp_input, rel_cell], dim=-1)

                query_count = coord.shape[1]
                pred = self.imnet(mlp_input.view(bs * query_count, -1)).view(
                    bs, query_count, -1)
                preds.append(pred)
                areas.append(torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1]) + 1e-9)

        total_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            areas[0], areas[3] = areas[3], areas[0]
            areas[1], areas[2] = areas[2], areas[1]

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / total_area).unsqueeze(-1)
        return ret.view(bs, height, width, -1).permute(0, 3, 1, 2)

    def forward(self, inp):
        h, w = inp.shape[2], inp.shape[3]
        b = inp.shape[0]
        coord = make_coord((h, w), device=inp.device)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        cell = cell.unsqueeze(0).repeat(b, 1, 1)
        coord = coord.unsqueeze(0).repeat(b, 1, 1)
        points_enc = self.positional_encoding(coord, levels=POSITION_ENCODING_LEVELS)
        coord = torch.cat([coord, points_enc], dim=-1)
        return self.query(inp, coord, cell)

    def positional_encoding(self, x, levels):
        freq = 2 ** torch.arange(levels, dtype=torch.float32, device=x.device) * np.pi
        spectrum = x[..., None] * freq
        sin, cos = spectrum.sin(), spectrum.cos()
        return torch.stack([sin, cos], dim=-2).view(*x.shape[:-1], -1)


class INR2(INR):
    """Coordinate MLP branch used for three-channel RGB reconstruction."""

    def __init__(self, dim, local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__(
            dim,
            out_dim=3,
            local_ensemble=local_ensemble,
            feat_unfold=feat_unfold,
            cell_decode=cell_decode,
        )


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1,
                              padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class MINRMamba(nn.Module):
    """MINR-Mamba image deraining network."""

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        bias=False,
        dual_pixel_task=False,
    ):
        super().__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = self._make_stage(dim, num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = self._make_stage(int(dim * 2), num_blocks[1])
        self.down2_3 = Downsample(int(dim * 2))
        self.encoder_level3 = self._make_stage(int(dim * 4), num_blocks[2])
        self.down3_4 = Downsample(int(dim * 4))
        self.latent = self._make_stage(int(dim * 8), num_blocks[3])

        self.up4_3 = Upsample(int(dim * 8))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 8), int(dim * 4), kernel_size=1, bias=bias)
        self.decoder_level3 = self._make_stage(int(dim * 4), num_blocks[2])

        self.up3_2 = Upsample(int(dim * 4))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 4), int(dim * 2), kernel_size=1, bias=bias)
        self.decoder_level2 = self._make_stage(int(dim * 2), num_blocks[1])

        self.up2_1 = Upsample(int(dim * 2))
        self.decoder_level1 = self._make_stage(int(dim * 2), num_blocks[0])
        self.refinement = self._make_stage(int(dim * 2), num_refinement_blocks)

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2), kernel_size=1, bias=bias)
        self.output = nn.Conv2d(int(dim * 2), out_channels, kernel_size=3,
                                stride=1, padding=1, bias=bias)

        # Keep these attribute names for compatibility with existing checkpoints.
        self.hsv = nn.Conv2d(1, 48, kernel_size=1, bias=bias)
        self.hsv2 = nn.Conv2d(3, 48, kernel_size=1, bias=bias)
        self.INR = INR(dim=48)
        self.INR2 = INR2(dim=48)

    @staticmethod
    def _make_stage(dim, depth):
        return nn.Sequential(*[
            LFSSBlock(
                dim,
                d_state=8,
                inner_rank=32,
                num_tokens=64,
                convffn_kernel_size=5,
                mlp_ratio=1.,
            )
            for _ in range(depth)
        ])

    def forward(self, inp_img):
        hsv = rgb_to_hsv_batch(inp_img)
        hsv_v = self.hsv(hsv[:, 2:3, :, :])
        v_process = self.INR(hsv_v)

        hsv_enhanced = hsv.clone()
        hsv_enhanced[:, 2:3, :, :] = v_process
        hsv_rgb = hsv_to_rgb_batch(hsv_enhanced) + inp_img

        rgb_inr = self.hsv2(inp_img)
        rgb_inr = self.INR2(rgb_inr) + inp_img
        x = inp_img + hsv_rgb * 0.05 + rgb_inr * 0.05

        enc1 = self.encoder_level1(self.patch_embed(x))
        enc2 = self.encoder_level2(self.down1_2(enc1))
        enc3 = self.encoder_level3(self.down2_3(enc2))
        latent = self.latent(self.down3_4(enc3))

        dec3 = self.up4_3(latent)
        dec3 = self.reduce_chan_level3(torch.cat([dec3, enc3], 1))
        dec3 = self.decoder_level3(dec3)

        dec2 = self.up3_2(dec3)
        dec2 = self.reduce_chan_level2(torch.cat([dec2, enc2], 1))
        dec2 = self.decoder_level2(dec2)

        dec1 = self.up2_1(dec2)
        dec1 = self.decoder_level1(torch.cat([dec1, enc1], 1))
        dec1 = self.refinement(dec1)

        if self.dual_pixel_task:
            dec1 = dec1 + self.skip_conv(enc1)
            return self.output(dec1)
        return self.output(dec1) + inp_img
