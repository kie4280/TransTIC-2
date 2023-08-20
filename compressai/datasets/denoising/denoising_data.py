from PIL import Image
from glob import glob

from torch.utils.data import Dataset
from torch import Tensor
import torchvision.transforms as transforms
import torch

from typing import Literal, List

DATASETS = Literal["BSD400", "DIV2K", "Flickr2K", "WaterlooED", "flicker"]


class GaussianNoise(Dataset):
    def __init__(
        self,
        base_dir: str = "./",
        sigma:float = 15,
        transform = None,
        datasets: List[DATASETS] = [
            "BSD400", "DIV2K", "Flickr2K", "WaterlooED", "flicker"],
    ) -> None:
        super().__init__()
        self.base_dir = base_dir
        files = []
        for d in datasets:
            g = glob(f"{self.base_dir}/{d}/**/*.jpg", recursive=True)
            files.extend(g)
            g = glob(f"{self.base_dir}/{d}/**/*.png", recursive=True)
            files.extend(g)
        print(f"{len(files)} images loaded")
        self.files = files
        self.sigma = sigma
        self.transform = transform
        self.default_trans = transforms.ToTensor()
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> [Tensor, Tensor]:
        image = Image.open(self.files[index])
        image = self.default_trans(image)
        
        if self.transform != None:
           image = self.transform(image) 
        
        gaussian = torch.randn_like(image)
        lq = image + gaussian / 255.0 * self.sigma
        
        return lq, image


class RealNoise(Dataset):
    def __init__(self, base_dir: str = "./") -> None:
        super().__init__()
        self.base_dir = base_dir

    def __len__(self):
        pass

    def __getitem__(self, index) -> [Tensor, Tensor]:
        pass

if __name__ == "__main__":
    ga = GaussianNoise("/disk2/dataset")
    print(ga[0])