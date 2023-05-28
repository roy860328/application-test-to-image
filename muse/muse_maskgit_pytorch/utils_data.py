"""
pip install datasets

"""
import os
import argparse
from datasets import load_dataset
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class ImageDescDataset(Dataset):
    def __init__(self, 
                 dataset_name, 
                 image_size, 
                 use_min_data=False,
                 is_train=True,
                 is_only_image=False):
        """
        dataset format
            DatasetDict({
                train: Dataset({
                    features: ['image', 'description', 'label', 'file_name'],
                    num_rows: 5994
                })
                test: Dataset({
                    features: ['image', 'description', 'label', 'file_name'],
                    num_rows: 5794
                })
            })
        """
        self.dataset = load_dataset(dataset_name)
        self.dataset = self.dataset["train"] if is_train else self.dataset["test"]
        self.dataset_image = self.dataset["image"]
        self.dataset_description = self.dataset["description"]
        self.n = len(self.dataset) if not use_min_data else 3
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor(),
                                             transforms.Lambda(lambda img: img.repeat(3, 1, 1) if img.shape[0] == 1 else img)])
        self.is_only_image = is_only_image

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = self.dataset_image[idx]
        image = self.transform(image)
        description = self.dataset_description[idx]
        description = self._random_choie(description)
        if self.is_only_image:
            return image
        else:
            return image, description

    def _random_choie(self, description):
        # random choice one sentence
        description = [sentence for sentence in description.split("\n") if sentence != ""]
        # overfitting
        # description = description[np.random.choice(len(description))]
        description = description[0]
        return description

def get_args():
    parser = argparse.ArgumentParser(description='Dataset script')
    parser.add_argument('--dataset', type=str, default="CUB-200")
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_train_steps_vae', type=int, default=200)
    parser.add_argument('--path_save_vae', type=str, default="./model/model_vae.pt")
    parser.add_argument('--path_save_base', type=str, default="./model/model_base.pt")
    parser.add_argument('--path_save_superres', type=str, default="./model/model_superres.pt")
    parser.add_argument('--use_min_data', type=bool, default=True)

    args, unknown = parser.parse_known_args()
    return args

# dataset
def get_dataset(args, image_size, is_train, is_only_image):
    if args.dataset == "CUB-200":
        dataset_CUB = "alkzar90/CC6204-Hackaton-Cub-Dataset"
        dataset = ImageDescDataset(dataset_name=dataset_CUB,
                                   image_size=image_size,
                                   use_min_data=args.use_min_data,
                                   is_train=is_train,
                                   is_only_image=is_only_image)
    else:
        raise "not implemented"
    
    return dataset

# just for check utils_data.py is working
def test():
    args = get_args()
    dataset = get_dataset(args, image_size=args.image_size, is_train=True, is_only_image=False)
    # hyper parameter
    batch_size = 2

    train_loader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=batch_size, 
                                               shuffle=False,
                                               drop_last=True,
                                               num_workers=0)

    for idx, (img, description) in enumerate(train_loader):
        # just check dataset is working
        # plt img / print description
        from matplotlib import pyplot as plt
        plt.imshow(img[0].permute(1, 2, 0))
        plt.show()
        print(description)
        if idx == 4:
            break

