from torch.utils.data import Dataset
from typing import Literal
import numpy as np
from PIL import Image

GaussianBlurType = Literal["Urban100"]

class GaussianBlurSyn(Dataset):
    def __init__(self, dataset_name:GaussianBlurType="Urban100", 
                 transform=None) -> None:
        self.dataset_name:GaussianBlurType = dataset_name
        self.transform = transform

    def __getitem__(self, index) -> np.numpy:
        img = Image.open()

    def __len__(self):
        if self.dataset_name == "Urban100":
            pass