import torch
import numpy as np
import argparse
import yaml
import os
from tqdm import tqdm
import socket 
import datetime 

from torch.utils.data import DataLoader

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))

from src.prediction.prediction_network import PredictionLSTM
from src.prediction.prediction import evaluate_loss
from src.dataset.sequencedataset import LatentSequenceDataset
from src.utils.other_utils import load_config

from src.utils.visualize_grid import image_grid, save_image_grid, visualize_prediction
from src.utils.writer import Writer

now = datetime.datetime.now()

seed = 123
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

HOST = socket.gethostname()

dataset = 'Nuscenes'

if dataset == 'Waymo':
    step = 1
else:
    step = 2

MODEL_NAME = f'{dataset}-lstm-prediction-dynamics-engine'
DATE = now.strftime("%d-%m-%Y-%H-%M-%S") #today.strftime("%b-%d-%Y")
BATCH_SIZE = 512


config = {
    "nb_epochs": 2000,
    "samples_per_epoch": 100, 
    "batch_size": BATCH_SIZE,
    "checkpoint_interval": 10,
    "num_workers": 40,
    "z_c_recon_coeff": 1,
    "z_t_recon_coeff": 0.01
}

config["DynamicsEngine"] = {
    "extrap_t": 5,
    "nt": 20,
    "model": {
        "split": 128,
        "fc_model": [
            ['Linear', {'in': 128, 'out': 128}],
            ['Activation', 'leaky_relu']
        ],
        "conv_model": [
           # ['Conv2D_BN', {'in': 64, 'out': 128, 'bias': True, 'kernel': 1, 'stride': 1, 'activation': 'leaky_relu'}],
            #['Conv2D_BN', {'in': 128, 'out': 128, 'bias': True, 'kernel': 2, 'stride': 1, 'activation': 'leaky_relu'}],
            ['Conv2D_BN', {'in': 64, 'out': 128, 'bias': True, 'kernel': 3, 'stride': 1, 'padding': 1, 'activation': 'leaky_relu'}], #(512,4,4)
            ['Conv2D_BN', {'in': 128, 'out': 896, 'bias': True, 'kernel': 4, 'stride': 1, 'padding': 0, 'activation': 'leaky_relu'}], #(512,4,4)
           # ['AvgPool2d', {'in': 128, 'out': 128, 'kernel': 5}], #O: 512
            ['Linear', {'in': 896, 'out': 896}]
        ],
        "recurrent_model": [
            ["LSTM", {"input_size": 1024, "hidden_size": 1024, "num_layers": 3, "batch_first": True}]
        ],
        "conv_content_model": [
            ['Conv2D_BN', {'in': 1024 - 128, 'out': 64, 'bias': True, 'kernel': 1, 'stride': 1, 'padding': 0, 'activation': 'none'}], #(512,4,4)
        ],
    }

}



config["dataset"] = {
    "datafile": f'/home/benksy/Projects/Datasets/LatentVAE{dataset}DatasetAugust2022',
    "nt": 20,
    "mode": "all",
    "step": step
}

nb_epochs = config["nb_epochs"]
samples_per_epoch = config["samples_per_epoch"]
batch_size = config["batch_size"]
checkpoint_interval = config["checkpoint_interval"]
num_workers = config["num_workers"]
length = len(config['dataset']['datafile'])
log_dir = f'outputs/{HOST}-{MODEL_NAME}-{length}-{DATE}'
os.mkdir(log_dir)

with open(f'{log_dir}/config.yml', 'w') as outfile:
    yaml.dump(config, outfile)

writer = Writer(log_dir, f'{MODEL_NAME}-{DATE}', config)
train_set = LatentSequenceDataset(config["dataset"], 'train')
val_set = LatentSequenceDataset(config["dataset"], 'val')
train_dataloader = MultiEpochsDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
val_dataloader = MultiEpochsDataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
data_sample_train = (next(iter(train_dataloader)))

print(f'Data sample:{data_sample_train[0].shape, data_sample_train[1].shape}')
model = PredictionLSTM(config).cuda()

print(model)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print(f'Number of params:{params}')

optimizer = torch.optim.Adam(model.parameters(), betas=(0.9,0.98), eps=1e-09)

train_loss = []
val_loss = []
total_idx = 0
best_val = np.Inf
for epoch in tqdm(range(nb_epochs)):
    train_loss_per_epoch = []
    idx = 0
    for idx, x in enumerate(train_dataloader):
        total_idx += 1
        if idx<samples_per_epoch:
            #Check if needed
            z_theme = x[0].cuda()
            z_content = x[1].cuda()
            #print(f'{z_theme.shape, z_content.shape}')
            loss, logs = evaluate_loss(z_theme, z_content, model, None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_logs(logs, total_idx, 'train')
            train_loss_per_epoch.append(loss.item())
        else:
            val_loss_per_epoch = 0
            model.eval()
            val_loss_per_epoch = []
            logs = {}
            with torch.no_grad():
                for idx, x in enumerate(val_dataloader):
                    if idx<10*samples_per_epoch:
                        #Check if needed
                        z_theme = x[0].float().cuda()
                        z_content = x[1].float().cuda()
                        loss, logs = evaluate_loss(z_theme, z_content, model, logs)
                        val_loss_per_epoch.append(loss.item())
                    else:
                        break

                for key in logs.keys():
                    logs[key] = logs[key]/(idx+1)
                writer.add_logs(logs, total_idx, 'val')

                val_loss_per_epoch = np.mean(val_loss_per_epoch)

                if val_loss_per_epoch < best_val:
                    best_val = val_loss_per_epoch
                    print("Saving the best model at epoch {}".format(epoch))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        }, f'{log_dir}/ckpt_best.pt')

            model.train()
            train_loss_per_epoch = np.mean(train_loss_per_epoch)

            print(f'Epoch: {epoch}/{nb_epochs}, Training loss:{train_loss_per_epoch}, Validation loss:{val_loss_per_epoch}')
            writer.add_scalar('train_loss', train_loss_per_epoch, epoch)
            writer.add_scalar('val_loss', val_loss_per_epoch, epoch)
            train_loss.append(train_loss_per_epoch)
            val_loss.append(val_loss_per_epoch)
            #idx += 1
            break

    if epoch%checkpoint_interval == 0:
        print("Saving the model at epoch {}".format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            }, f'{log_dir}/ckpt_{epoch}.pt')

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    }, f'{log_dir}/final.pt')

writer.close()
