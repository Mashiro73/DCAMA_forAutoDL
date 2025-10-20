import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# ❗️ 核心依赖：确保这些导入在您的环境中有效 ❗️
# selective_scan_cuda_oflex 需要通过编译 kernels/selective_scan 安装
# csm_triton 需要将 csm_triton.py 复制到同一目录下，并安装 triton
try:
    import selective_scan_cuda_oflex # 底层 CUDA 实现
except ImportError:
    print("WARNING: selective_scan_cuda_oflex not found. VSSM might not work correctly.")
    selective_scan_cuda_oflex = None # 或者提供一个纯 Python 的回退实现 (如果有)

try:
    from .csm_triton import CrossScanTriton, CrossMergeTriton # Triton 加速实现
    has_triton = True
except ImportError:
    print("WARNING: csm_triton not found or triton not installed. VSSM will use slower PyTorch scan.")
    has_triton = False
    CrossScanTriton = None
    CrossMergeTriton = None

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


def toodd(size):
    size = to_2tuple(size)
    if size[0] % 2 == 1:
        pass
    else:
        size[0] = size[0] + 1
    if size[1] % 2 == 1:
        pass
    else:
        size[1] = size[0] + 1
    return size

class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

class VSSM(nn.Module):

    def __init__(
            self,
            d_model=96,
            d_state=1,
            expansion_ratio=1,
            dt_rank="auto",
            norm_layer=LayerNorm2d,
            dropout=0.0,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            k_groups=4,
            **kwargs,
    ):

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(expansion_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # # # out proj =======================================
        # self.out_norm = norm_layer(d_inner)

        self.expansion_ratio = expansion_ratio

        if self.expansion_ratio != 1.0:
            self.proj = nn.Linear(d_model, d_inner)
            self.yproj = nn.Linear(d_inner, d_model)

        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0).view(-1, d_inner, 1))
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_groups, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_groups, merge=True)  # (K * D)

        # self.factor1 = nn.Parameter(torch.ones(d_inner, 1, 1), requires_grad=True)
        # self.factor2 = nn.Parameter(torch.ones(d_inner, 1, 1), requires_grad=True)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def _selective_scan(self, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=None,
                        backnrows=None, ssoflex=False):
        return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    def _cross_scan(self, x):
        return CrossScanTriton.apply(x)

    def _cross_merge(self, x):
        return CrossMergeTriton.apply(x)

    def forward(self, x, to_dtype=False, force_fp32=False):
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        # xs = torch.stack([x, x.flip([-1])], dim=1).reshape(B, -1, L)
        # xs = self._cross_scan(x).reshape(B, -1, L)
        if self.expansion_ratio != 1.0:
            x = self.proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            xs = self._cross_scan(x)  # size b, 4, embed_dim, L
            xs = xs.reshape(B, -1, L)
        else:
            xs = self._cross_scan(x).reshape(B, -1, L)
        x_dbl = F.conv1d(xs, self.x_proj_weight, bias=None, groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), dt_projs_weight.reshape(K * D, -1, 1), groups=K)

        dts = dts.contiguous().reshape(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().reshape(B, K, N, L)
        Cs = Cs.contiguous().reshape(B, K, N, L)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.reshape(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)

        ys = self._selective_scan(xs, dts, As, Bs,
                                  Cs, Ds, delta_bias,
                                  delta_softplus=True,
                                  ssoflex=True)

        y = self._cross_merge(ys.reshape(B, K, -1, H, W)).reshape(B, -1, L)

        if self.expansion_ratio != 1.0:
            y = self.yproj(y.permute(0, 2, 1)).permute(0, 2, 1)

        if to_dtype:
            y = y.to(x.dtype)

        return y
