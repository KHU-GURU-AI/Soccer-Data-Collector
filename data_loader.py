import os

from PIL import Image
from torch.utils.data import Dataset

"""
Real image dataset
"""
class RealDataset(Dataset):
    def __init__(self, real_image_dir):
        self.real_images = sorted([os.path.join(real_image_dir, file_name) \
                                  for file_name in os.listdir(real_image_dir)])

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, idx):
        image = Image.open(self.real_images[idx])
        return image

"""
Synthetic dataset
"""
class SyntheticDataset(Dataset):
    def __init__(self, syn_image_dir, syn_depth_dir):
        self.syn_images = sorted([os.path.join(syn_image_dir, file_name) \
                                  for file_name in os.listdir(syn_image_dir)])
        self.syn_depths = sorted([os.path.join(syn_depth_dir, file_name) \
                                  for file_name in os.listdir(syn_depth_dir)])

    def __len__(self):
        return len(self.syn_images)

    def __getitem__(self, idx):
        image = Image.open(self.syn_images[idx])
        depth = Image.open(self.syn_depths[idx])
        return image, depth
