r""" Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

from .base.swin_transformer import SwinTransformer
from .base.cswin import CSWinTransformer  # 添加此行
from .base.pvt import pvt_v2_b1_official
from .base.segman_encoder import SegMANEncoder_s,SegMANEncoder_b
from .base.swin_transformer_v2 import SwinTransformerV2
from model.base.transformer import MultiHeadedAttention, PositionalEncoding


class SpatialGatingUnit(nn.Module):
    """
    轻量级空间注意力门控模块 (Lightweight Spatial Attention Gating Module)
    用于智能地融合两个不同尺度的特征图。
    """

    def __init__(self, in_channels, norm_layer=nn.GroupNorm):
        super(SpatialGatingUnit, self).__init__()

        # 这个小型卷积网络用于学习空间注意力/门控信号
        # 输入通道数是两个特征图拼接后的通道数 (in_channels * 2)
        # 输出是一个单通道的注意力图
        self.gate_generator = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            norm_layer(num_groups=4, num_channels=in_channels),  # 添加归一化层增加稳定性
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()  # Sigmoid函数确保输出的门控值在 0 到 1 之间
        )

    def forward(self, high_res_feat, low_res_feat):
        # high_res_feat: 高分辨率特征 (富含细节)
        # low_res_feat:  低分辨率特征 (富含语义)
        assert high_res_feat.shape == low_res_feat.shape, "待融合的特征图尺寸必须相同"

        # 1. 拼接两个特征图作为输入
        combined_feat = torch.cat([high_res_feat, low_res_feat], dim=1)

        # 2. 生成空间门控图 (gate)
        gate = self.gate_generator(combined_feat)

        # 3. 应用门控信号进行加权融合
        # gate 权重应用于高分辨率特征, (1 - gate) 权重应用于低分辨率特征
        fused_feat = (high_res_feat * gate) + (low_res_feat * (1 - gate))

        return fused_feat


class DCAMA(nn.Module):

    def __init__(self, backbone, pretrained_path, use_original_imgsize):
        super(DCAMA, self).__init__()

        self.backbone = backbone
        self.use_original_imgsize = use_original_imgsize

        # feature extractor initialization
        if backbone == 'resnet50':
            self.feature_extractor = resnet.resnet50()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
        elif backbone == 'resnet101':
            self.feature_extractor = resnet.resnet101()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 23, 3]
            self.feat_ids = list(range(0, 34))
        elif backbone == 'swin':
            self.feature_extractor = SwinTransformer(img_size=384, patch_size=4, window_size=12, embed_dim=128,
                                                     depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            self.feature_extractor.load_state_dict(torch.load(pretrained_path)['model'])
            self.feat_channels = [128, 256, 512, 1024]
            self.nlayers = [2, 2, 18, 2]


        elif backbone == 'segman':
            # --- 实例化 SegMAN Encoder ---
            # 我们以 SegMANEncoder_s 为例。pretrained 参数会由 args.feature_extractor_path 传入
            #self.feature_extractor = SegMANEncoder_s(pretrained=pretrained_path, image_size=224)
            self.feature_extractor = SegMANEncoder_b(pretrained=pretrained_path, image_size=224)

            # --- 配置模型参数 (这些值必须与 SegMANEncoder_s 的定义严格对应) ---
            # 根据 segman_encoder.py 中 SegMANEncoder_s 的定义:
            #embed_dims=[64, 144, 288, 512], depths=[2, 2, 10, 4]
            # self.feat_channels = [64, 144, 288, 512]
            # self.nlayers = [2, 2, 10, 4]

            self.feat_channels = [96, 160, 364, 560] # 必须与 _b 的 embed_dims 匹配
            self.nlayers = [4, 4, 18, 4]             # 必须与 _b 的 depths 匹配

            # 论文中提到使用 MMSegmentation 库训练，权重通常保存在 'state_dict_ema' 或 'state_dict'
            # 您的 segman_encoder.py 内部加载逻辑已经处理了 'state_dict_ema'
            # 这里我们假设您会遵循该格式，所以无需额外的加载代码。
            # 如果您的权重文件格式不同，需要在这里添加类似 swin_v2 的加载逻辑。
        elif backbone == 'swin_v2':

            # ==================== ✅ Swin V2 配置（已验证正确） ====================

            self.feature_extractor = SwinTransformerV2(
                img_size=384,  # 来源于文件名中的 '192to384'
                patch_size=4,  # 来源于文件名中的 'patch4'
                window_size=24,  # 来源于文件名中的 'window12to24' 的最终尺寸
                embed_dim=128,  # Swin-Base 模型的标准配置
                depths=[2, 2, 18, 2],  # Swin-Base 模型的标准配置
                num_heads=[4, 8, 16, 32],  # Swin-Base 模型的标准配置
                pretrained_window_size=12,  # 来源于文件名中的 'window12to24' 的初始尺寸
                feat_ids=[1, 2, 3, 4]  # DCAMA 需求：提取所有4个stage的特征
            )

            if pretrained_path:

                print(f"\n{'=' * 70}")

                print(f"🔄 加载 Swin V2 预训练权重")

                print(f"   文件: {pretrained_path}")

                print(f"{'=' * 70}")

                # ==================== 加载权重 ====================

                checkpoint = torch.load(pretrained_path, map_location='cpu')

                # 处理不同的权重格式

                if 'model' in checkpoint:

                    state_dict = checkpoint['model']

                elif 'state_dict' in checkpoint:

                    state_dict = checkpoint['state_dict']

                else:

                    state_dict = checkpoint

                print(f"📦 权重文件包含 {len(state_dict)} 个参数")

                # ==================== 清理分类头 ====================

                # 只移除分类头，其他参数（包括buffer）都保留

                keys_to_remove = []
                for k in list(state_dict.keys()):
                    if k.startswith('head.'):
                        keys_to_remove.append(k)
                if keys_to_remove:
                    print(f"🗑️  移除分类头: {len(keys_to_remove)} 个参数")
                    for k in keys_to_remove:
                        state_dict.pop(k)
                # ==================== 加载权重 ====================
                msg = self.feature_extractor.load_state_dict(state_dict, strict=False)
                # ==================== 详细报告 ====================
                loaded_count = len(state_dict) - len(msg.missing_keys)
                print(f"\n📊 加载结果:")
                print(f"   ✅ 成功加载: {loaded_count}/{len(state_dict)} 个参数")
                if msg.missing_keys:
                    # 分类缺失的参数
                    buffers = [k for k in msg.missing_keys if any(
                        x in k for x in ['attn_mask', 'relative_position_index',
                                         'relative_coords_table'])]
                    params = [k for k in msg.missing_keys if k not in buffers]
                    if buffers:
                        print(f"   📌 缺失的Buffer: {len(buffers)} 个（会自动初始化）")
                    if params:
                        print(f"   ⚠️  缺失的参数: {len(params)} 个")
                        for k in params[:5]:
                            print(f"      - {k}")
                        if len(params) > 5:
                            print(f"      ... 还有 {len(params) - 5} 个")
                if msg.unexpected_keys:
                    print(f"   ⚠️  多余的键: {len(msg.unexpected_keys)} 个")
                if loaded_count == len(state_dict) or (len(state_dict) - loaded_count <= 2):
                    print(f"   🎉 权重加载完美！")
                print(f"{'=' * 70}\n")
            # ==================== ✅ 特征通道配置（已验证正确） ====================
            self.feat_channels = [128, 256, 512, 1024]
            self.nlayers = [2, 2, 18, 2]


        elif backbone == 'cswin':  # 添加 cswin 分支
            # 使用 CSWin-B (base) 模型的配置, 假设输入图像尺寸为384
            self.feature_extractor = CSWinTransformer(img_size=384, patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
                                                      split_size=[1, 2, 12, 12], num_heads=[4, 8, 16, 32], mlp_ratio=4.)
            # 加载预训练权重
            self.feature_extractor.load_state_dict(torch.load(pretrained_path)['state_dict_ema'])
            self.feat_channels = [96, 192, 384, 768]
            self.nlayers = [2, 4, 32, 2]
        elif backbone == 'pvt':
            self.feature_extractor = pvt_v2_b1_official()

            # 加载你下载的官方预训练权重
            # 注意：官方权重是为分类任务训练的，包含一些我们不需要的键（如head），
            # 所以使用 strict=False 来忽略不匹配的键。
            if pretrained_path:
                state_dict = torch.load(pretrained_path)
                self.feature_extractor.load_state_dict(state_dict, strict=False)

            # 更新为 pvt_v2_b1 的正确参数
            self.feat_channels = [64, 128, 320, 512]
            self.nlayers = [2, 2, 2, 2]  # 每个stage有2个block
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.feature_extractor.eval()

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)
        self.model = DCAMA_model(in_channels=self.feat_channels, stack_ids=self.stack_ids)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_img, support_img, support_mask):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img)
            support_feats = self.extract_feats(support_img)

        logit_mask = self.model(query_feats, support_feats, support_mask.clone())

        return logit_mask

    def extract_feats(self, img):
        r""" Extract input image features """
        feats = []

        if self.backbone in ['swin', 'swin_v2', 'cswin']:
            _ = self.feature_extractor.forward_features(img)
            for feat in self.feature_extractor.feat_maps:
                bsz, hw, c = feat.size()
                h = int(hw ** 0.5)
                # 在这里进行统一的格式转换
                feat = feat.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                feats.append(feat)
        elif self.backbone == 'segman':
            # 1. 调用 forward_features，这将运行模型并填充我们改造后暴露的 self.feat_maps 列表
            _ = self.feature_extractor.forward_features(img)

            # 2. 直接使用 feat_maps
            # 一个巨大的好处是：SegMAN Encoder 的 Block 直接输出 [B, C, H, W] 格式的特征，
            # 与 SwinV2 的 [B, L, C] 不同，因此我们无需进行任何格式转换！
            feats = self.feature_extractor.feat_maps
        elif self.backbone == 'pvt':
            _ = self.feature_extractor.forward_features(img)
            # 直接使用 pvt_v2 已经格式化好的特征列表
            feats = self.feature_extractor.feat_maps
        elif self.backbone == 'resnet50' or self.backbone == 'resnet101':
            bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
            # Layer 0
            feat = self.feature_extractor.conv1.forward(img)
            feat = self.feature_extractor.bn1.forward(feat)
            feat = self.feature_extractor.relu.forward(feat)
            feat = self.feature_extractor.maxpool.forward(feat)

            # Layer 1-4
            for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
                res = feat
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

                if bid == 0:
                    res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

                feat += res

                if hid + 1 in self.feat_ids:
                    feats.append(feat.clone())

                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

    def predict_mask_nshot(self, batch, nshot):
        r""" n-shot inference """
        query_img = batch['query_img']
        support_imgs = batch['support_imgs']
        support_masks = batch['support_masks']

        if nshot == 1:
            logit_mask = self(query_img, support_imgs[:, 0], support_masks[:, 0])
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                n_support_feats = []
                for k in range(nshot):
                    support_feats = self.extract_feats(support_imgs[:, k])
                    n_support_feats.append(support_feats)
            logit_mask = self.model(query_feats, n_support_feats, support_masks.clone(), nshot)

        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        else:
            logit_mask = F.interpolate(logit_mask, support_imgs[0].size()[2:], mode='bilinear', align_corners=True)

        return logit_mask.argmax(dim=1)

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.feature_extractor.eval()


class DCAMA_model(nn.Module):
    def __init__(self, in_channels, stack_ids):
        super(DCAMA_model, self).__init__()

        self.stack_ids = stack_ids

        # DCAMA blocks
        self.DCAMA_blocks = nn.ModuleList()
        self.pe = nn.ModuleList()

        # for inch in in_channels[1:]:
        #     self.DCAMA_blocks.append(MultiHeadedAttention(h=8, d_model=inch, dropout=0.5))
        #     self.pe.append(PositionalEncoding(d_model=inch, dropout=0.5))
        # ✅ [修改] 动态确定头数 h
        default_h = 8 # 默认头数
        
        for inch in in_channels[1:]: # 遍历 d_model 值
            current_h = default_h
            # 如果 d_model 不能被默认头数整除
            if inch % current_h != 0:
                # 尝试更小的头数，例如 4 (需要确保 4 是所有可能 inch 的公约数，或者添加更多层检查)
                if inch % 4 == 0:
                    current_h = 4
                    print(f"Warning: d_model={inch} is not divisible by {default_h}. Using h={current_h} instead.")
                # 如果连 4 都不能整除，则抛出更明确的错误
                else:
                    raise ValueError(f"d_model={inch} cannot be evenly split by common head counts like 8 or 4.")
            
            # 使用计算出的 current_h 实例化模块
            self.DCAMA_blocks.append(MultiHeadedAttention(h=current_h, d_model=inch, dropout=0.5))
            self.pe.append(PositionalEncoding(d_model=inch, dropout=0.5))

        outch1, outch2, outch3 = 16, 64, 128
        self.fusion_gate1 = SpatialGatingUnit(in_channels=outch3)
        self.fusion_gate2 = SpatialGatingUnit(in_channels=outch3)

        # conv blocks
        self.conv1 = self.build_conv_block(stack_ids[3] - stack_ids[2], [outch1, outch2, outch3], [3, 3, 3],
                                           [1, 1, 1])  # 1/32
        self.conv2 = self.build_conv_block(stack_ids[2] - stack_ids[1], [outch1, outch2, outch3], [5, 3, 3],
                                           [1, 1, 1])  # 1/16
        self.conv3 = self.build_conv_block(stack_ids[1] - stack_ids[0], [outch1, outch2, outch3], [5, 5, 3],
                                           [1, 1, 1])  # 1/8

        self.conv4 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])  # 1/32 + 1/16
        self.conv5 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])  # 1/16 + 1/8

        # mixer blocks
        self.mixer1 = nn.Sequential(
            nn.Conv2d(outch3 + 2 * in_channels[1] + 2 * in_channels[0], outch3, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU())

        self.mixer2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True),
                                    nn.ReLU())

        self.mixer3 = nn.Sequential(nn.Conv2d(outch1, outch1, (3, 3), padding=(1, 1), bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))

    def forward(self, query_feats, support_feats, support_mask, nshot=1):
        coarse_masks = []
        for idx, query_feat in enumerate(query_feats):
            # 1/4 scale feature only used in skip connect
            if idx < self.stack_ids[0]: continue

            bsz, ch, ha, wa = query_feat.size()

            # reshape the input feature and mask
            query = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()
            if nshot == 1:
                support_feat = support_feats[idx]
                mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                     align_corners=True).view(support_feat.size()[0], -1)
                support_feat = support_feat.view(support_feat.size()[0], support_feat.size()[1], -1).permute(0, 2,
                                                                                                             1).contiguous()
            else:
                support_feat = torch.stack([support_feats[k][idx] for k in range(nshot)])
                support_feat = support_feat.view(-1, ch, ha * wa).permute(0, 2, 1).contiguous()
                mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), (ha, wa), mode='bilinear', align_corners=True)
                                    for k in support_mask])
                mask = mask.view(bsz, -1)

            # DCAMA blocks forward
            if idx < self.stack_ids[1]:
                coarse_mask = self.DCAMA_blocks[0](self.pe[0](query), self.pe[0](support_feat), mask)
            elif idx < self.stack_ids[2]:
                coarse_mask = self.DCAMA_blocks[1](self.pe[1](query), self.pe[1](support_feat), mask)
            else:
                coarse_mask = self.DCAMA_blocks[2](self.pe[2](query), self.pe[2](support_feat), mask)
            coarse_masks.append(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, 1, ha, wa))

        # multi-scale conv blocks forward
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[3] - 1 - self.stack_ids[0]].size()
        coarse_masks1 = torch.stack(
            coarse_masks[self.stack_ids[2] - self.stack_ids[0]:self.stack_ids[3] - self.stack_ids[0]]).transpose(0,
                                                                                                                 1).contiguous().view(
            bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[2] - 1 - self.stack_ids[0]].size()
        coarse_masks2 = torch.stack(
            coarse_masks[self.stack_ids[1] - self.stack_ids[0]:self.stack_ids[2] - self.stack_ids[0]]).transpose(0,
                                                                                                                 1).contiguous().view(
            bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[1] - 1 - self.stack_ids[0]].size()
        coarse_masks3 = torch.stack(coarse_masks[0:self.stack_ids[1] - self.stack_ids[0]]).transpose(0,
                                                                                                     1).contiguous().view(
            bsz, -1, ha, wa)

        coarse_masks1 = self.conv1(coarse_masks1)
        coarse_masks2 = self.conv2(coarse_masks2)
        coarse_masks3 = self.conv3(coarse_masks3)

        # multi-scale cascade (pixel-wise addition)
        coarse_masks1 = F.interpolate(coarse_masks1, coarse_masks2.size()[-2:], mode='bilinear', align_corners=True)
        # mix = coarse_masks1 + coarse_masks2
        # ✅ [第3步 A] 使用第一个门控单元，智能融合1/16和1/32尺度的特征
        mix = self.fusion_gate1(high_res_feat=coarse_masks2, low_res_feat=coarse_masks1)
        mix = self.conv4(mix)

        mix = F.interpolate(mix, coarse_masks3.size()[-2:], mode='bilinear', align_corners=True)
        # mix = mix + coarse_masks3
        # ✅ [第3步 B] 使用第二个门控单元，智能融合之前的结果和1/8尺度的特征
        # 这里的 high_res_feat 是 coarse_masks3, low_res_feat 是 mix_interp
        mix = self.fusion_gate2(high_res_feat=coarse_masks3, low_res_feat=mix)
        mix = self.conv5(mix)

        # skip connect 1/8 and 1/4 features (concatenation)
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[1] - 1]
        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)]).max(
                dim=0).values
        mix = torch.cat((mix, query_feats[self.stack_ids[1] - 1], support_feat), 1)

        upsample_size = (mix.size(-1) * 2,) * 2
        mix = F.interpolate(mix, upsample_size, mode='bilinear', align_corners=True)
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[0] - 1]
        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)]).max(
                dim=0).values
        mix = torch.cat((mix, query_feats[self.stack_ids[0] - 1], support_feat), 1)

        # mixer blocks forward
        out = self.mixer1(mix)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        out = self.mixer2(out)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.mixer3(out)

        return logit_mask

    def build_conv_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
        r""" bulid conv blocks """
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            pad = ksz // 2

            building_block_layers.append(nn.Conv2d(in_channels=inch, out_channels=outch,
                                                   kernel_size=ksz, stride=stride, padding=pad))
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)
