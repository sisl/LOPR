import argparse
import torch
from torch.utils import data
import torchvision
import os
import sys
from tqdm import tqdm
import pathlib


sys.path.append(str(pathlib.Path(__file__).parents[1]))
from src.utils.load_styleVAE import load_styleVAE
from src.dataset.single_frame_dataset import OGMImageDataset
import pickle

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ogm_dataset_path', type=str, default='/home/benksy/Projects/Datasets/NuscenesImgsAugust2022StyleGAN')
    parser.add_argument('--latent_dataset_path', type=str, default='/home/benksy/Projects/Datasets/LatentVAENuscenesDatasetAugust2022')
    parser.add_argument('--ckpt_path', type=str, default='/home/benksy/Projects/CoRL/trained_models/Nuscenes_DriveGAN/270000.pt')

    parser.add_argument('--iter', type=int, default=800000)
    parser.add_argument('--save_iter', type=int, default=10000)

    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--n_sample', type=int, default=6)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--gamma', type=float, default=50.0)

    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--ckpt', type=str, default=None)

    parser.add_argument('--log_dir', type=str, default='./results6')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='carla')
    parser.add_argument('--n_mlp', type=int, default=8)
    parser.add_argument('--constant_input_size', type=int, default=4)
    parser.add_argument('--num_patchD', type=int, default=0)
    parser.add_argument('--theme_dim', type=int, default=128)
    parser.add_argument('--spatial_dim', type=int, default=4)
    parser.add_argument('--spatial_beta', type=float, default=2.0)
    parser.add_argument('--theme_beta', type=float, default=1.0)

    parser.add_argument('--width_mul', type=float, default=1)
    parser.add_argument('--crop_input', type=int, default=0)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--theme_spat_dim', type=int, default=32)
    args = parser.parse_args()

    dataset = 'carla' #Why?
    size = 128
    batch = 128

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    config = {
        'ckpt': args.ckpt_path, 
        'size': size,
        'n_mlp': 8,
        'device': 'cuda',
        'channel_multiplier': 2,
        'args': args
    }

    vae = load_styleVAE(config)
    vae.eval()
    print('Loaded the models')

    def convert_dataset(loader, mode):
        """
        Converts the image dataset to latent vector dataset
        """
        for batch_ndx, sample in tqdm(enumerate(iter(loader))):
            real_img = sample[0].cuda()
            fns = sample[1]

            return_latent_only = True

            out = vae(real_img, replace_input={}, decode_only=False, return_latent_only=return_latent_only, train=False)
            out['real_img'] = real_img.cpu().numpy()

            # Save latents.
            for idx, fn in enumerate(fns):
                scene_id, frame = fn.split('/')[-2:]
                path = f'{args.latent_dataset_path}/{mode}/{scene_id}'
                if not os.path.isdir(path):
                    os.mkdir(path)

                outfile = f'{path}/{frame[:-4]}'
                data = {'spatial_mu': out['spatial_mu'][idx].cpu().numpy(), 'theme_mu': out['theme_mu'][idx].cpu().numpy()}
                with open(outfile, 'wb') as f:
                    pickle.dump(data, f)

    train_dataset = OGMImageDataset(args.ogm_dataset_path+'/train', args.dataset, args.size, train=False, args=args, with_scene_info=True)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=256,
    )
    val_dataset = OGMImageDataset(args.ogm_dataset_path+'/val', args.dataset, args.size, train=False, args=args, with_scene_info=True)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=256,
    )

    test_dataset = OGMImageDataset(args.ogm_dataset_path+'/test', args.dataset, args.size, train=False, args=args, with_scene_info=True)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=256,
    )

    os.mkdir(os.path.join(args.latent_dataset_path, 'train'))
    os.mkdir(os.path.join(args.latent_dataset_path, 'val'))
    os.mkdir(os.path.join(args.latent_dataset_path, 'test'))

    with torch.no_grad():
        convert_dataset(train_loader, 'train')
        convert_dataset(val_loader, 'val')
        convert_dataset(test_loader, 'test')

        