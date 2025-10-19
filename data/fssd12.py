# DCAMA/data/fssd12.py (Fixed Version)

import os
import random
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data


class DatasetFSSD12(data.Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        # Store arguments
        self.split = split
        self.benchmark = 'fssd12'  # ✅ Required for AverageMeter
        self.shot = shot
        self.fold = fold
        self.use_original_imgsize = use_original_imgsize
        self.datapath = datapath
        self.transform = transform

        # Define paths
        self.img_path_root = os.path.join(datapath, 'FSSD-12')

        # Build metadata
        self.class_names = self._build_class_names()

        # ✅ FIXED: Add nclass attribute (required by AverageMeter)
        # nclass is the number of classes in the current fold
        self.nclass = len(self.class_names)

        # ✅ FIXED: Add class_ids attribute (required by AverageMeter)
        # class_ids is a list of indices for the current fold's classes
        self.class_ids = list(range(self.nclass))

        self.img_metadata_classwise = self._build_img_metadata_classwise()
        self.img_metadata = self._build_img_metadata()  # ✅ Flat list for indexing

    def __len__(self):
        # ✅ Return actual number of images
        return len(self.img_metadata)

    def __getitem__(self, idx):
        # Sample one episode (a query image and k-shot support images from the same class)
        # ✅ FIXED: Now receives both class_sample (int) and class_name (str)
        query_name, support_names, class_sample, class_name = self.sample_episode(idx)

        # Load images and masks
        # ✅ FIXED: Pass class_name (string) to load_frame
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names, class_name)

        # ✅ Apply transformation to query image first
        query_img = self.transform(query_img)

        # ✅ Resize query mask to match transformed image size using interpolate
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(),
                                   query_img.size()[-2:], mode='nearest').squeeze()

        # ✅ Apply transformation to support images
        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        # ✅ Resize support masks to match transformed image size
        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(),
                                  support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {
            'query_img': query_img,
            'query_mask': query_mask,
            'query_name': query_name,
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'support_names': support_names,
            'class_id': torch.tensor(class_sample),  # ✅ Return as tensor
        }
        return batch

    def _build_class_names(self):
        """Reads the class names from the split file for the current fold"""
        split_path = os.path.join('data', 'splits', 'fssd12', self.split, f'fold{self.fold}.txt')
        with open(split_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names

    def _build_img_metadata_classwise(self):
        """Organizes a dictionary where keys are class names and values are lists of image names"""
        metadata_classwise = {class_name: [] for class_name in self.class_names}
        for class_name in self.class_names:
            img_dir = os.path.join(self.img_path_root, class_name, 'Images')
            if os.path.isdir(img_dir):
                img_names = [img_name for img_name in os.listdir(img_dir) if img_name.endswith('.jpg')]
                metadata_classwise[class_name] = img_names
        return metadata_classwise

    def _build_img_metadata(self):
        """Build a flat list of all (class_name, img_name) tuples for indexing"""
        img_metadata = []
        for class_name in self.class_names:
            for img_name in self.img_metadata_classwise[class_name]:
                img_metadata.append((class_name, img_name))
        return img_metadata

    def sample_episode(self, idx):
        """Sample an episode: choose query image and support images from the same class"""
        # Get the query image from the index
        class_name, query_name = self.img_metadata[idx]
        class_sample = self.class_names.index(class_name)

        # Get all images for this class
        class_images = self.img_metadata_classwise[class_name]

        # Sample support images (ensure they're different from query)
        support_names = []
        available_images = [img for img in class_images if img != query_name]

        if len(available_images) >= self.shot:
            support_names = random.sample(available_images, self.shot)
        else:
            # If not enough images, sample with replacement
            support_names = random.choices(available_images if available_images else class_images, k=self.shot)

        # ✅ FIXED: Return both class_sample (int) and class_name (str)
        return query_name, support_names, class_sample, class_name

    def load_frame(self, query_name, support_names, class_name):
        """Load images and masks, return as PIL Images and torch tensors

        Args:
            query_name: image filename (e.g., 'image001.jpg')
            support_names: list of support image filenames
            class_name: class name string (e.g., 'cat', 'dog')
        """
        # ✅ FIXED: Now class_name is correctly used as a string
        # Construct full paths to the query image and its mask
        query_img_path = os.path.join(self.img_path_root, class_name, 'Images', query_name)
        query_mask_path = os.path.join(self.img_path_root, class_name, 'GT', query_name.replace('.jpg', '.png'))

        # Load the query image and mask
        query_img = Image.open(query_img_path).convert('RGB')
        query_mask = self.read_mask(query_mask_path)

        # Load support images and masks
        support_imgs = []
        support_masks = []
        for name in support_names:
            s_img_path = os.path.join(self.img_path_root, class_name, 'Images', name)
            s_mask_path = os.path.join(self.img_path_root, class_name, 'GT', name.replace('.jpg', '.png'))

            support_imgs.append(Image.open(s_img_path).convert('RGB'))
            support_masks.append(self.read_mask(s_mask_path))

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, mask_path):
        """Read mask and convert to binary tensor (0 or 1)"""
        mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
        # Binarize the mask
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask