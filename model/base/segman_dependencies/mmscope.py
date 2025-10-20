# 文件路径: model/base/segman_dependencies/mmscope.py
# 方案三 (最终修复版)：将 MMSCopE 重构为通用的三尺度融合解码器

import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ 从同级目录导入 VSSM 和 LayerNorm2d
from .vssm_utils import VSSM, LayerNorm2d

# 使用 F.interpolate 作为 resize 函数
resize = F.interpolate

class MMSCopE(nn.Module):
    """
    MMSCopE (Mamba-based Multi-Scale Context Extraction) 解码器模块。
    
    此版本被修改为接收来自 DCAMA 的三个不同尺度的粗略掩码 (1/8, 1/16, 1/32)，
    使用 VSSM (Mamba) 进行融合，并输出 1/8 尺度的精炼特征图。
    """
    def __init__(self, embed_dim, norm_layer=LayerNorm2d, act_layer=nn.GELU):
        """
        Args:
            embed_dim (int): 融合模块的内部通道维度。
                             在 DCAMA 中，这将是 `outch3` (例如 128)。
        """
        super().__init__()
        self.embed_dim = embed_dim
        intermediate_dim = self.embed_dim

        # --- 1. 定义 PixelUnshuffle 操作 ---
        self.unshuffle_f = nn.PixelUnshuffle(4)
        self.unshuffle_s2 = nn.PixelUnshuffle(2)

        # --- 2. 定义三个尺度的通道投影器 ---
        self.reduce_channels = nn.ModuleList([
            # [0] 用于 1/8 (unshuffled, C*16) -> intermediate_dim
            nn.Sequential(
                nn.Conv2d(self.embed_dim * 16, intermediate_dim, kernel_size=1),
                norm_layer(intermediate_dim), act_layer()
            ),
            # [1] 用于 1/16 (unshuffled, C*4) -> intermediate_dim
            nn.Sequential(
                nn.Conv2d(self.embed_dim * 4, intermediate_dim, kernel_size=1),
                norm_layer(intermediate_dim), act_layer()
            ),
            # [2] 用于 1/32 (C) -> intermediate_dim
            nn.Sequential(
                nn.Conv2d(self.embed_dim, intermediate_dim, kernel_size=1),
                norm_layer(intermediate_dim), act_layer()
            )
        ])

        # --- 3. VSSM (Mamba) 模块 ---
        # 输入通道为 3 * intermediate_dim (三个尺度拼接后)
        self.vssm = VSSM(d_model=intermediate_dim * 3)

        # --- 4. 输出投影 ---
        # 输入通道 (intermediate_dim * 3) 必须与 VSSM 的 d_model 匹配
        # 输出通道为 embed_dim (128)
        self.proj_out = nn.Sequential(
            nn.Conv2d(intermediate_dim * 3, self.embed_dim, kernel_size=1)
        )
        self.norm_out = norm_layer(self.embed_dim)

    def forward(self, x_8, x_16, x_32):
        """
        Args:
            x_8 (torch.Tensor): 1/8 尺度的粗略掩码 (来自 conv3)
                                Shape: [B, C, H, W]
            x_16 (torch.Tensor): 1/16 尺度的粗略掩码 (来自 conv2)
                                 Shape: [B, C, H/2, W/2]
            x_32 (torch.Tensor): 1/32 尺度的粗略掩码 (来自 conv1)
                                 Shape: [B, C, H/4, W/4]
        Returns:
            torch.Tensor: Mamba 融合后的特征图，恢复到 1/8 尺度
                          Shape: [B, C, H, W]
        """
        # 1. Pixel Unshuffle：全部统一到 1/32 尺度
        x_unshuffled = self.unshuffle_f(x_8)
        s2_unshuffled = self.unshuffle_s2(x_16)
        s4_unshuffled = x_32

        # 2. 通道投影
        x_reduced = self.reduce_channels[0](x_unshuffled)
        s2_reduced = self.reduce_channels[1](s2_unshuffled)
        s4_reduced = self.reduce_channels[2](s4_unshuffled)

        # 3. 拼接 (Concatenate)
        # (B, 3 * C_inter, H/4, W/4)
        x_concat = torch.cat([x_reduced, s2_reduced, s4_reduced], dim=1)
        
        # ✅ [关键修复] -----------------------------------------------
        # 3.5 获取 4D 形状，VSSM 会将其压缩为 3D
        B, C_concat, H_prime, W_prime = x_concat.shape

        # 4. VSSM (Mamba) 扫描
        # 输入是 4D: [B, C_concat, H', W']
        # VSSM.forward 的输出是 3D: [B, C_concat, L']  (其中 L' = H' * W')
        x_vssm_3d = self.vssm(x_concat) 

        # 4.5 [关键修复] 将 3D 输出 reshape 回 4D，以便 Conv2d (proj_out) 可以处理
        # (B, C_concat, L') -> (B, C_concat, H', W')
        x_vssm_4d = x_vssm_3d.view(B, C_concat, H_prime, W_prime)
        # --------------------------------------------------------

        # 5. 输出投影
        # (B, C_embed, H/4, W/4)
        x_proj = self.norm_out(self.proj_out(x_vssm_4d))

        # 6. 恢复尺度：将融合后的特征图上采样回 1/8 尺度
        # (B, C_embed, H, W)
        x_out = resize(x_proj, size=x_8.shape[-2:], mode='bilinear', align_corners=False)
        
        return x_out