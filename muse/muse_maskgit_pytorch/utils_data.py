"""
pip install datasets

"""
import os
import argparse
from datasets import load_dataset

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class ImageDescDataset(Dataset):
    def __init__(self, 
                 dataset_name, 
                 image_size, 
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
        self.n = len(self.dataset)
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor()])
        self.is_only_image = is_only_image

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = self.dataset_image[idx]
        image = self.transform(image)
        description = self.dataset_description[idx]
        if self.is_only_image:
            return image
        else:
            return image, description

def get_args():
    parser = argparse.ArgumentParser(description='Dataset script')
    parser.add_argument('--dataset', type=str, default="CUB-200")
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size_vae', type=int, default=2)
    parser.add_argument('--num_train_steps_vae', type=int, default=10)
    parser.add_argument('--path_save_vae', type=str, default="./mode/vae.pt")
    parser.add_argument('--path_save_base', type=str, default="./mode/base.pt")
    parser.add_argument('--path_save_superres', type=str, default="./mode/superres.pt")
    args, unknown = parser.parse_known_args()
    return args

# dataset
def get_dataset(args, is_train, is_only_image):
    if args.dataset == "CUB-200":
        dataset_CUB = "alkzar90/CC6204-Hackaton-Cub-Dataset"
        dataset = ImageDescDataset(dataset_name=dataset_CUB,
                                   image_size=args.image_size,
                                   is_train=is_train,
                                   is_only_image=is_only_image)
    else:
        raise "not implemented"
    
    return dataset

# just for check utils_data.py is working
def test():
    args = get_args()
    dataset = get_dataset(args, is_train=True, is_only_image=False)
    # hyper parameter
    batch_size = 2

    train_loader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True,
                                               num_workers=2)

    for idx, (img, description) in enumerate(train_loader):
        # just check dataset is working
        # plt img / print description
        from matplotlib import pyplot as plt
        plt.imshow(img[0].permute(1, 2, 0))
        plt.show()
        print(description)
        if idx == 4:
            break

