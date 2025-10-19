r""" Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation """
import torch.nn as nn
import torch

from model.DCAMA import DCAMA
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset


def test(model, dataloader, nshot):
    r""" Test """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. forward pass
        batch = utils.to_cuda(batch)
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  iou_b=area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    args = parse_opts()

    Logger.initialize(args, training=False)

    # Model initialization
    model = DCAMA(args.backbone, args.feature_extractor_path, args.use_original_imgsize)
    model.eval()

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    params = model.state_dict()
    state_dict = torch.load(args.load)

    if not args.load:
        raise Exception('Pretrained model not specified.')

    print(f"--- 正在从以下路径加载最终模型权重: {args.load} ---")

    # 使用 weights_only=True 更安全，并能消除PyTorch的警告
    # map_location 确保模型能被加载，无论它是在CPU还是GPU上保存的
    state_dict = torch.load(args.load, map_location=device, weights_only=True)

    # 直接加载 state_dict。
    # 因为训练和测试都使用了 nn.DataParallel, 权重键的前缀 ('module.') 应该是匹配的。
    # 我们不再需要那个危险的 for 循环。
    try:
        model.load_state_dict(state_dict)
        print("--- 模型权重加载成功！ ---")
    except RuntimeError as e:
        print("\n--- 错误：模型结构与加载的权重不匹配。 ---")
        print("--- 这通常发生在模型的保存和加载状态不一致（例如，一个带 'module.' 前缀，一个不带）。")
        print(f"--- 原始报错信息: {e}\n")
        # 如果直接加载失败，可以尝试手动移除 'module.' 前缀（作为备用方案）
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # 从加载的权重中移除 'module.'
            new_state_dict[name] = v
        print("--- 尝试移除 'module.' 前缀后再次加载... ---")
        model.load_state_dict(new_state_dict)
        print("--- 模型权重（移除前缀后）加载成功！ ---")

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize, args.vispath)

    # Dataset initialization
    FSSDataset.initialize(img_size=384, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    # Test
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
