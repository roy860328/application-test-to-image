import os
import torch
import torch.nn.functional as F
from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer
from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer

# set
vq_codebook_size = 512
dim_vae = 256
layer = 2

seq_len = 256
dim_transformer = 512
dim_depth = 8
dim_dim_head = 64
dim_heads = 8

def init_vae(args):
    vae = VQGanVAE(
        dim = dim_vae,
        vq_codebook_size = vq_codebook_size,
        layer = layer
    )
    if os.path.exists(args.path_save_vae):
        vae.load(args.path_save_vae)
    return vae

def init_transformer(is_base=False):
    transformer = MaskGitTransformer(
        num_tokens = vq_codebook_size,         # must be same as codebook size above
        seq_len = seq_len if is_base else seq_len*4,            # must be equivalent to fmap_size ** 2 in vae
        dim = dim_transformer,                # model dimension
        depth = dim_depth,                # depth
        dim_head = dim_dim_head,            # attention head dimension
        heads = dim_heads,                # attention heads,
        ff_mult = 4,              # feedforward expansion factor
        t5_name = 't5-small',     # name of your T5
    )
    return transformer

def init_base(args, vae, transformer):
    base_maskgit = MaskGit(
        vae = vae,                 # vqgan vae
        transformer = transformer, # transformer
        image_size = args.image_size,          # image size
        cond_drop_prob = 0.25,     # conditional dropout, for classifier free guidance
    )
    if os.path.exists(args.path_save_base):
        base_maskgit.load(args.path_save_base)
    return base_maskgit

def init_superres(args, vae, transformer):
    superres_maskgit = MaskGit(
        vae = vae,
        transformer = transformer,
        cond_drop_prob = 0.25,
        image_size = args.image_size*2,                     # larger image size
        cond_image_size = args.image_size,                # conditioning image size <- this must be set
    )
    if os.path.exists(args.path_save_superres):
        superres_maskgit.load(args.path_save_superres)
    return superres_maskgit

def train_vae(args):
    vae = init_vae(args)

    dataset = utils_data.get_dataset(args, 
                                     image_size=args.image_size, 
                                     is_train=True, 
                                     is_only_image=True)
    
    trainer = VQGanVAETrainer(
        vae = vae,
        image_size = args.image_size,             # you may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it
        folder = '**deprecated**',
        dataset=dataset,
        batch_size = args.batch_size,
        grad_accum_every = 8,
        num_train_steps = args.num_train_steps_vae
    )
    trainer.train()
    if not os.path.exists(args.path_save_vae):
        os.makedirs(args.path_save_vae)
    vae.save(args.path_save_vae)

def train_base(args, dataset):
    vae = init_vae(args)
    transformer = init_transformer(is_base=True)
    base_maskgit = init_base(args, vae, transformer)

    dataset = utils_data.get_dataset(args, 
                                     image_size=args.image_size, 
                                     is_train=True, 
                                     is_only_image=False)
    train_loader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=0)
    for epoch in range(args.num_epochs):
        for i, (images, texts) in enumerate(train_loader):
            loss = base_maskgit(images, 
                                texts=texts)
            loss.backward()
            base_maskgit.step()
            base_maskgit.zero_grad()
            if base_maskgit.step() % 100 == 0:
                print('step: %d, loss: %.5f' % (base_maskgit.step(), loss.item()))

    base_maskgit.save(args.path_save_base)

def train_superres(args):
    vae = init_vae(args)
    transformer = init_transformer(is_base=False)
    superres_maskgit = init_superres(args, vae, transformer)


    dataset = utils_data.get_dataset(args, 
                                     image_size=args.image_size*2, 
                                     is_train=True, 
                                     is_only_image=False)

    # similiar with train_base
    train_loader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=0)
    for epoch in range(args.num_epochs):
        for i, (images, texts) in enumerate(train_loader):
            loss = superres_maskgit(images, 
                                    texts=texts)
            loss.backward()
            superres_maskgit.step()
            superres_maskgit.zero_grad()
            if superres_maskgit.step() % 100 == 0:
                print('step: %d, loss: %.5f' % (superres_maskgit.step(), loss.item()))

    superres_maskgit.save(args.path_save_superres)

def inference(args):
    from muse_maskgit_pytorch import Muse
    vae = init_vae(args)
    transformer = init_transformer(is_base=True)
    base_maskgit = init_base(args, vae, transformer)
    transformer = init_transformer(is_base=False)
    superres_maskgit = init_superres(args, vae, transformer)

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
    from muse_maskgit_pytorch import utils_data
    args = utils_data.get_args()    
    train_vae(args)
    train_base(args)
    train_superres(args)

    inference()
