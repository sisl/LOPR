"""
Converts occupancy grids to latent vectors using a trained StyleGAN network. 
Code by Bernard Lange
"""
import argparse
import torch
from torch.utils import data
import torchvision
import os
import sys
from drivegan_code_modified_old.latent_decoder_model.dataset import BernardImageDatasetRaw
from load_styleVAE import load_styleVAE
from tqdm import tqdm
import cv2
import pathlib
import numpy as np
import hickle as hkl
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.dataset.sequencedataset import LatentSequenceDataset, SequenceDataset, SequenceDatasetDriveGAN
from src.utils.visualize_grid import save_image_grid
import pickle

torch.backends.cudnn.benchmark = True

def save_img(name, data, n_sample, scale=True):
    sample = data[:n_sample]
    if scale:
        sample = sample * 0.5 + 0.5
    sample = torch.clamp(sample, 0, 1.0)
    x = torchvision.utils.make_grid(
        sample, nrow=int(n_sample ** 0.5),
        normalize=False, scale_each=False
    )
    torchvision.utils.save_image(x, name)
    #logger.add_image(name, x, step)

def save_img_sequence(name, data, n_sample, scale=True):
    #sample = data[:n_sample]
    N, T, C, W, H = data.shape
    sample = data.reshape(N*T,C,W,H)

    if scale:
        sample = sample * 0.5 + 0.5
    sample = torch.clamp(sample, 0, 1.0)
    x = torchvision.utils.make_grid(
        sample, nrow=T,
        normalize=False, scale_each=False
    )
    torchvision.utils.save_image(x, name)
    #logger.add_image(name, x, step)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='/home/benksy/Projects/Datasets/NuscenesImgsAugust2022')
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
        'ckpt': '/home/benksy/Projects/CoRL/results_vae/240000.pt', #'/home/benksy/Projects/CoRL_Rebuttal/Nuscenes/270000.pt', #'/home/benksy/Projects/CoRL/results_new_dataset/230000.pt', # '/home/benksy/Projects/CoRL/results_vae/290000.pt',
        'size': size,
        'n_mlp': 8,
        'device': 'cuda',
        'channel_multiplier': 2,
        'args': args
    }

    vae = load_styleVAE(config)
    vae.eval()
    print('Loaded the models')

    def convert_dataset(loader, mode, randomly_visualize=False):
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
                #print('\n \n  This is: ',fn, '\n \n')
                scene_id, frame = fn.split('/')[-2:]
                # print(f'{scene_id}, {frame}')
                path = f'{latent_dataset_path}/{mode}/{scene_id}'
                if not os.path.isdir(path):
                    os.mkdir(path)

                outfile = f'{path}/{frame[:-4]}'
                #hkl.dump({'spatial_mu': out['spatial_mu'][idx].cpu().numpy(), 'theme_mu': out['theme_mu'][idx].cpu().numpy()}, outfile, mode='w')
                data = {'spatial_mu': out['spatial_mu'][idx].cpu().numpy(), 'theme_mu': out['theme_mu'][idx].cpu().numpy()}
                with open(outfile, 'wb') as f:
                    pickle.dump(data, f)


    train_dataset = BernardImageDatasetRaw(args.path+'/train', args.dataset, args.size, every_second=True, train=False, args=args)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=256,
    )
    val_dataset = BernardImageDatasetRaw(args.path+'/val', args.dataset, args.size, every_second=True, train=False, args=args)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=256,
    )

    test_dataset = BernardImageDatasetRaw(args.path+'/test', args.dataset, args.size, every_second=True, train=False, args=args)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=256,
    )

    latent_dataset_path = '/home/benksy/Projects/Datasets/LatentVAENuscenesDatasetAugust2022'
    # os.mkdir(os.path.join(latent_dataset_path, 'train'))
    # os.mkdir(os.path.join(latent_dataset_path, 'val'))
    # os.mkdir(os.path.join(latent_dataset_path, 'test'))
    # os.mkdir(os.path.join(latent_dataset_path, 'vis'))

    with torch.no_grad():
        convert_dataset(train_loader, 'train', randomly_visualize=True)
        convert_dataset(val_loader, 'val', randomly_visualize=True)
        convert_dataset(test_loader, 'test', randomly_visualize=True)

        