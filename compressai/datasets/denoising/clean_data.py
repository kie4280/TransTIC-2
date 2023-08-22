from PIL import Image
from glob import glob

from torch.utils.data import Dataset
from torch import Tensor
import torchvision.transforms as transforms
import torch

from typing import Literal, List
import numpy as np
from tqdm import tqdm

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
        image = np.asarray(image)
        print(image.shape)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if image.shape[2] == 1:
            ni = np.zeros((image.shape[0], image.shape[1], 3))
            print(image.shape)
            for i in range(3):
                ni [:, :, i] = image[:, :,0]
            image = ni
            Image.fromarray(image, "RGB").save(self.files[index])
            
        elif image.shape[2] == 4:
            print(image.shape)
            image = image[:, :, 0:3]
            Image.fromarray(image, "RGB").save(self.files[index])
        elif image.shape[2] == 3:
            pass
        else:
            raise ValueError("invalid dimension")
        
        

        # print(image.shape)

        return image


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
    args.training_dataset = ["WaterlooED"]
    args.testing_data_path = "/disk2/dataset/test"
    args.testing_dataset = ["Urban100"]
    ga = GaussianNoise(args, "test")

    for i in tqdm(range(len(ga))):
        img = ga[i]
        print(img.shape)