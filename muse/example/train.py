
from muse_maskgit_pytorch import utils_data
args = utils_data.get_args
dataset = utils_data.get_dataset(args, is_train=True, is_only_image=True)

import torch
import torch.nn.functional as F
from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer
from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer

def init_vae():
    vae = VQGanVAE(
        dim = 256,
        vq_codebook_size = 512
    )
    return vae

def init_base_transformer():
    transformer = MaskGitTransformer(
        num_tokens = 512,         # must be same as codebook size above
        seq_len = 256,            # must be equivalent to fmap_size ** 2 in vae
        dim = 512,                # model dimension
        depth = 8,                # depth
        dim_head = 64,            # attention head dimension
        heads = 8,                # attention heads,
        ff_mult = 4,              # feedforward expansion factor
        t5_name = 't5-small',     # name of your T5
    )
    return transformer

def init_superres_transformer():
    transformer = MaskGitTransformer(
        num_tokens = 512,         # must be same as codebook size above
        seq_len = 1024,           # must be equivalent to fmap_size ** 2 in vae
        dim = 512,                # model dimension
        depth = 2,                # depth
        dim_head = 64,            # attention head dimension
        heads = 8,                # attention heads,
        ff_mult = 4,              # feedforward expansion factor
        t5_name = 't5-small',     # name of your T5
    )
    return transformer

def init_base(vae, transformer):
    base_maskgit = MaskGit(
        vae = vae,                 # vqgan vae
        transformer = transformer, # transformer
        image_size = 256,          # image size
        cond_drop_prob = 0.25,     # conditional dropout, for classifier free guidance
    )
    return base_maskgit

def init_superres(vae, transformer):
    superres_maskgit = MaskGit(
        vae = vae,
        transformer = transformer,
        cond_drop_prob = 0.25,
        image_size = 512,                     # larger image size
        cond_image_size = 256,                # conditioning image size <- this must be set
    )
    return superres_maskgit

def train_vae():
    vae = init_vae()

    trainer = VQGanVAETrainer(
        vae = vae,
        image_size = args.image_size,             # you may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it
        folder = '**deprecated**',
        dataset=dataset,
        batch_size = args.batch_size_vae,
        grad_accum_every = 8,
        num_train_steps = args.num_train_steps_vae
    )
    trainer.train()
    vae.save(args.path_save_vae)

def train_base():
    # first instantiate your vae

    vae = init_vae()

    vae.load(args.path_save_vae) # you will want to load the exponentially moving averaged VAE

    # then you plug the vae and transformer into your MaskGit as so

    # (1) create your transformer / attention network

    transformer = init_base_transformer()

    # (2) pass your trained VAE and the base transformer to MaskGit

    base_maskgit = init_base(vae, transformer)

    # ready your training text and images

    texts = [
        'a child screaming at finding a worm within a half-eaten apple',
        'lizard running across the desert on two feet',
        'waking up to a psychedelic landscape',
        'seashells sparkling in the shallow waters'
    ]

    images = torch.randn(4, 3, 256, 256).cuda()

    # feed it into your maskgit instance, with return_loss set to True

    loss = base_maskgit(
        images,
        texts = texts
    )

    loss.backward()

    # do this for a long time on much data
    # then...

    images = base_maskgit.generate(texts = [
        'a whale breaching from afar',
        'young girl blowing out candles on her birthday cake',
        'fireworks with blue and green sparkles'
    ], cond_scale = 3.) # conditioning scale for classifier free guidance

    images.shape # (3, 3, 256, 256)
    
    base_maskgit.save(args.path_save_base)

def train_superres():
    # first instantiate your ViT VQGan VAE
    # a VQGan VAE made of transformers

    vae = init_vae()

    vae.load(args.path_save_vae) # you will want to load the exponentially moving averaged VAE

    # then you plug the VqGan VAE into your MaskGit as so

    # (1) create your transformer / attention network

    transformer = init_superres_transformer()

    # (2) pass your trained VAE and the base transformer to MaskGit

    superres_maskgit = init_superres(vae, transformer)

    # ready your training text and images

    texts = [
        'a child screaming at finding a worm within a half-eaten apple',
        'lizard running across the desert on two feet',
        'waking up to a psychedelic landscape',
        'seashells sparkling in the shallow waters'
    ]

    images = torch.randn(4, 3, 512, 512).cuda()

    # feed it into your maskgit instance, with return_loss set to True

    loss = superres_maskgit(
        images,
        texts = texts
    )

    loss.backward()

    # do this for a long time on much data
    # then...

    images = superres_maskgit.generate(
        texts = [
            'a whale breaching from afar',
            'young girl blowing out candles on her birthday cake',
            'fireworks with blue and green sparkles',
            'waking up to a psychedelic landscape'
        ],
        cond_images = F.interpolate(images, 256),  # conditioning images must be passed in for generating from superres
        cond_scale = 3.
    )

    images.shape # (4, 3, 512, 512)
    superres_maskgit.save(args.path_save_superres)

def inference():
    from muse_maskgit_pytorch import Muse

    vae = init_vae()
    vae.load(args.path_save_vae) # you will want to load the exponentially moving averaged VAE

    transformer = init_base_transformer()
    base_maskgit = init_base(vae, transformer)

    transformer = init_superres_transformer()
    superres_maskgit = init_superres(vae, transformer)

    base_maskgit.load(args.path_save_base)
    superres_maskgit.load(args.path_save_superres)
    # pass in the trained base_maskgit and superres_maskgit from above

    muse = Muse(
        base = base_maskgit,
        superres = superres_maskgit
    )

    images = muse([
        'a whale breaching from afar',
        'young girl blowing out candles on her birthday cake',
        'fireworks with blue and green sparkles',
        'waking up to a psychedelic landscape'
    ])

    images # List[PIL.Image.Image]

if __name__ == '__main__':
    train_vae()
    train_base()
    train_superres()
    inference()