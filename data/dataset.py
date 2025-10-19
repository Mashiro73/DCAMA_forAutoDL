r""" Dataloader builder for few-shot semantic segmentation dataset  """
import torch.distributed as dist  # 1. 添加这一行导入
from torch.utils.data.distributed import DistributedSampler as Sampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torchvision import transforms

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.fss import DatasetFSS
from data.fssd12 import DatasetFSSD12 # 确保 fssd12.py 在同级目录下

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
            'fssd12': DatasetFSSD12, # 2. 确保这里有 fssd12
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):

        # 根据 benchmark 参数选择对应的数据集类
        dataset = cls.datasets[benchmark](datapath=cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot,
                                           use_original_imgsize=cls.use_original_imgsize)

        # 3. --- 关键修改部分 ---
        # 检查是否处于分布式（多GPU）训练模式
        is_distributed = dist.is_available() and dist.is_initialized()

        if is_distributed:
            # 如果是分布式训练，使用 DistributedSampler
            sampler = Sampler(dataset)
        else:
            # 如果不是分布式训练（我们的情况），训练时使用普通随机采样器来打乱数据
            # 验证或测试时则不需要采样器
            sampler = RandomSampler(dataset) if split == 'trn' else None

        # 在Windows上，为了稳定性，将 pin_memory 设置为 False
        # drop_last=True 对训练很重要，可以确保每个批次的大小都相同
        dataloader = DataLoader(dataset, batch_size=bsz, sampler=sampler, num_workers=nworker,
                                pin_memory=False, drop_last=True if split == 'trn' else False)
        # --- 修改结束 ---

        return dataloader