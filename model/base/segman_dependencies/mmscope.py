# 文件路径: model/base/segman_dependencies/mmscope.py
# 代码逻辑来源: 提取并改编自 segman_decoder.py 中的 SegMANHead

import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ 从同级目录导入 VSSM 和 LayerNorm2d
from .vssm_utils import VSSM, LayerNorm2d

# 使用 F.interpolate 作为 resize 函数
resize = F.interpolate

class MMSCopE(nn.Module):
    """
    Modular Mamba-based Multi-Scale Context Extraction Module.
    Extracted and adapted from SegMANHead for plug-and-play use.
    Replaces mmcv.ConvModule with standard nn.Sequential.
    """
    def __init__(self, embed_dim, norm_layer=LayerNorm2d, act_layer=nn.GELU):
        super().__init__()
        self.embed_dim = embed_dim
        # SegMAN 源码中 reduce_channels 输出通道是 C//2
        intermediate_dim = embed_dim // 2
        if intermediate_dim == 0:
             intermediate_dim = embed_dim # 防止 embed_dim=1 时出错

        # --- 1. 定义生成多尺度上下文的卷积层 (替换 ConvModule) ---
        self.conv_s2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            norm_layer(embed_dim),
            act_layer()
        )
        # 遵循 segman_decoder.py 源码的 stride=2 实现，输入为 s2
        self.conv_s4 = nn.Sequential(
             nn.Conv2d(embed_dim, embed_dim, kernel_size=5, stride=2, padding=2),
             norm_layer(embed_dim),
             act_layer()
        )

        # --- 2. 定义 PixelUnshuffle ---
        # 目标统一到 H/32 x W/32 (基于 conv_s4 的输出尺度)
        self.unshuffle_f = nn.PixelUnshuffle(4)  # x (H/8) -> H/32, C*16
        self.unshuffle_s2 = nn.PixelUnshuffle(2) # s2 (H/16) -> H/32, C*4
        # s4 (H/32) 无需 unshuffle

        # --- 3. 定义通道投影卷积 (替换 ConvModule) ---
        # 索引匹配 SegMAN 源码：[0] for s4, [1] for s2, [2] for x
        self.reduce_channels = nn.ModuleList([
             nn.Sequential( # for s4_unshuffled (input C)
                 nn.Conv2d(embed_dim, intermediate_dim, kernel_size=1),
                 norm_layer(intermediate_dim), act_layer()
             ),
             nn.Sequential( # for s2_unshuffled (input C*4)
                 nn.Conv2d(embed_dim * 4, intermediate_dim, kernel_size=1),
                 norm_layer(intermediate_dim), act_layer()
             ),
             nn.Sequential( # for f_unshuffled (input C*16)
                 nn.Conv2d(embed_dim * 16, intermediate_dim, kernel_size=1),
                 norm_layer(intermediate_dim), act_layer()
             )
        ])

        # --- 4. 实例化 Mamba 扫描模块 (VSSM) ---
        # 输入通道为 3 * intermediate_dim
        # 确保传入 use_triton 等参数（如果 vssm_utils.py 中的 VSSM 支持）
        self.vssm = VSSM(d_model=intermediate_dim * 3)

        # --- 5. 定义上采样和输出投影层 (替换 ConvModule) ---
        # SegMAN 源码直接 resize + proj_out
        self.proj_out = nn.Sequential(
            # 输入通道是 VSSM 输出的 3 * intermediate_dim
            nn.Conv2d(intermediate_dim * 3, embed_dim, kernel_size=1)
            # 输出投影后通常不接 Norm 和 Act
        )

        # --- 6. 定义输出归一化 (可选) ---
        self.norm_out = norm_layer(embed_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature map, expected to be at 1/8 scale.
                              Shape: [B, C, H, W], where C = self.embed_dim.
        Returns:
            torch.Tensor: Refined feature map with the same shape as input.
        """
        B, C, H, W = x.shape
        if C != self.embed_dim:
            raise ValueError(f"Input channel dimension ({C}) does not match MMSCopE embed_dim ({self.embed_dim})")

        # 1. 生成多尺度上下文表示
        s2 = self.conv_s2(x)  # [B, C, H/2, W/2] = 1/16 scale
        s4 = self.conv_s4(s2) # [B, C, H/4, W/4] = 1/32 scale (根据源码)

        # 2. Pixel Unshuffle 到 H/32 x W/32
        x_unshuffled = self.unshuffle_f(x)     # [B, C*16, H/4, W/4]
        s2_unshuffled = self.unshuffle_s2(s2)  # [B, C*4,  H/4, W/4]
        s4_unshuffled = s4                     # [B, C,    H/4, W/4]

        # 3. 通道投影 (映射到 intermediate_dim)
        # 索引调整: [0] for s4, [1] for s2, [2] for x
        f_proj  = self.reduce_channels[2](x_unshuffled)     # [B, C//2, H/4, W/4]
        s2_proj = self.reduce_channels[1](s2_unshuffled)    # [B, C//2, H/4, W/4]
        s4_proj = self.reduce_channels[0](s4_unshuffled)    # [B, C//2, H/4, W/4]

        # 4. 拼接
        concat_feat = torch.cat([f_proj, s2_proj, s4_proj], dim=1) # [B, 3*(C//2), H/4, W/4]

        # 5. Mamba 扫描 (VSSM)
        fused_feat = self.vssm(concat_feat) # 输出也是 [B, 3*(C//2), H/4, W/4]

        # 6. 恢复到 1/8 尺度 (使用 resize, 遵循源码)
        fused_feat_up = resize(fused_feat,
                               size=(H, W), # 恢复到 1/8 尺度 H, W
                               mode='bilinear',
                               align_corners=False) # 通常 align_corners=False for features

        # 7. 输出投影
        out_feat = self.proj_out(fused_feat_up) # [B, C, H, W]

        # 8. 输出归一化 (可选)
        out_feat = self.norm_out(out_feat)

        return out_feat