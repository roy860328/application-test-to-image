import os, sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from muse.example.train import init_vae, init_transformer, init_base, init_transformer, init_superres
from muse_maskgit_pytorch import Muse
from muse_maskgit_pytorch import utils_data

def get_model(args):
    vae = init_vae(args)
    transformer = init_transformer(is_base=True)
    base_maskgit = init_base(args, vae, transformer)
    transformer = init_transformer(is_base=False)
    superres_maskgit = init_superres(args, vae, transformer)

    muse = Muse(
        base = base_maskgit,
        superres = superres_maskgit
    )

    if torch.cuda.is_available():
        muse = muse.cuda()
    return muse

def inference(descriptions):
    assert type(descriptions) == type([]), "descriptions must be list"
    images = model(descriptions)
    return images

print("muse.utils_data start to loading muse model ...")
args = utils_data.get_args()
model = get_model(args)

print("muse.utils_data testing model work ...")
inference(["test model is fine"])