import os, sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__)))
from muse_maskgit_pytorch import utils_data

from matplotlib import pyplot as plt
import torchvision.transforms as T

def save_origin_dataset():
    args = utils_data.get_args()
    dataset = utils_data.get_dataset(args, 
                                     image_size=args.image_size, 
                                     is_train=True, 
                                     is_only_image=False)
    train_loader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=1, 
                                               shuffle=False,
                                               drop_last=True,
                                               num_workers=0)
    for i, (images, texts) in enumerate(train_loader):
        save_image_text(images, 
                        texts, 
                        path_name_image="dataset_image_origin", 
                        path_name_text="dataset_text_origin")

def save_image_text(images, texts, path_name_image="dataset_image_test", path_name_text="dataset_text_test"):
    path_image = f"./{path_name_image}"
    path_text = f"./{path_name_text}"
    if not os.path.exists(path_image):
        os.makedirs(path_image)
        os.makedirs(path_text)
    for i, (image, text) in enumerate(zip(images, texts)):
        image = T.ToPILImage()(image) if type(image) == torch.Tensor else image
        image.save(f"{path_image}/{i}.jpg")
        with open(f"{path_text}/{i}.txt", "w") as f:
            f.write(text)
