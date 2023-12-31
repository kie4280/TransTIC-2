from PIL import Image
from glob import glob

from torch.utils.data import Dataset
from torch import Tensor
import torchvision.transforms as transforms
import torch

from typing import Literal, List

class GaussianNoise(Dataset):
    def __init__(
        self,
        args,
        mode: Literal["train", "test"] = "train",
        sigma: float = 15,
        transform=None,
    ) -> None:
        super().__init__()
        base_dir = (
            args.training_data_path if mode == "train" else args.testing_data_path
        )
        datasets = args.training_dataset if mode == "train" else args.testing_dataset
        files = []
        for d in datasets:
            g = glob(f"{base_dir}/{d}/**/*.jpg", recursive=True)
            files.extend(g)
            g = glob(f"{base_dir}/{d}/**/*.png", recursive=True)
            files.extend(g)
            g = glob(f"{base_dir}/{d}/**/*.bmp", recursive=True)
            files.extend(g)
        print(f"{len(files)} images loaded")
        self.files = files
        self.sigma = sigma
        self.transform = transform
        self.mode = mode
        self.default_trans = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> [Tensor, Tensor]:
        image = Image.open(self.files[index])

        if self.transform != None:
            image = self.transform(image)
        else:
            image = self.default_trans(image)

        # print(image.shape)
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


class Object:
    pass
if __name__ == "__main__":
    
    args = Object()
    args.training_data_path = "/disk2/dataset/"
    args.training_dataset = ["flicker"]
    ga = GaussianNoise(args)

    for i, g in ga:
        if g.shape[0] != 3:
            print(g.shape)