r"""config"""
import argparse


def parse_opts():
    r"""arguments"""
    parser = argparse.ArgumentParser(
        description='Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation')

    # common
    parser.add_argument('--datapath', type=str, default='./datasets')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss', 'fssd12'])
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--bsz', type=int, default=2)
    parser.add_argument('--nworker', type=int, default=2)
    parser.add_argument('--backbone', type=str, default='segman',
                        choices=['resnet50', 'resnet101', 'swin', 'cswin', 'pvt', 'swin_v2', 'segman'])
    parser.add_argument('--feature_extractor_path', type=str, default='')
    parser.add_argument('--logpath', type=str, default='./logs')

    # for train
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--nepoch', type=int, default=300)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--finetune_backbone', action='store_true',
                        help='If set, the backbone network will be un-frozen and fine-tuned.')
    parser.add_argument('--use_amp', action='store_true', help='Enable Automatic Mixed Precision training')

    # for test
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--vispath', type=str, default='./vis')
    parser.add_argument('--use_original_imgsize', action='store_true')

    args = parser.parse_args()
    return args
