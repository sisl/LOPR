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
sys.path.append(str(pathlib.Path(__file__).parents[1]))

from src.prediction.prediction import PredictionLSTM
from src.prediction.prediction import evaluate_loss
from src.dataset.sequence_dataset import LatentSequenceDataset

from src.utils.writer import Writer
from src.utils.other_utils import MultiEpochsDataLoader

now = datetime.datetime.now()

seed = 123
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    from datetime import datetime
    date = datetime.now().isoformat(timespec='minutes')


    parser.add_argument('--path', type=str)
    parser.add_argument('--nt', type=int, default=20)
    parser.add_argument('--extrap_t', type=int, default=5)
    parser.add_argument('--nb_epochs', type=int, default=50)
    parser.add_argument('--samples_per_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=40)
    parser.add_argument('--z_c_recon_coeff', type=float, default=1)
    parser.add_argument('--z_t_recon_coeff', type=float, default=0.1)


    
    args = parser.parse_args()
    args.start_iter = 0

    HOST = socket.gethostname()

    dataset = 'Nuscenes'

    MODEL_NAME = f'lstm-prediction-dynamics-engine'
    DATE = now.strftime("%d-%m-%Y-%H-%M-%S")


    config = {
        "nb_epochs": args.nb_epochs,
        "samples_per_epoch": args.samples_per_epoch, 
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "z_c_recon_coeff": args.z_c_recon_coeff,
        "z_t_recon_coeff": args.z_t_recon_coeff
    }

    config["DynamicsEngine"] = {
        "extrap_t": args.extrap_t,
        "nt": args.nt,
        "model": {
            "split": 128,
            "fc_model": [
                ['Linear', {'in': 128, 'out': 128}],
                ['Activation', 'leaky_relu']
            ],
            "conv_model": [
                ['Conv2D_BN', {'in': 64, 'out': 128, 'bias': True, 'kernel': 3, 'stride': 1, 'padding': 1, 'activation': 'leaky_relu'}], 
                ['Conv2D_BN', {'in': 128, 'out': 896, 'bias': True, 'kernel': 4, 'stride': 1, 'padding': 0, 'activation': 'leaky_relu'}], 
                ['Linear', {'in': 896, 'out': 896}]
            ],
            "recurrent_model": [
                ["LSTM", {"input_size": 1024, "hidden_size": 1024, "num_layers": 3, "batch_first": True}]
            ],
            "conv_content_model": [
                ['Conv2D_BN', {'in': 1024 - 128, 'out': 64, 'bias': True, 'kernel': 1, 'stride': 1, 'padding': 0, 'activation': 'none'}], 
            ],
        }

    }



    config["dataset"] = {
        "datafile": args.path,
        "nt": args.nt,
        "mode": "all",
        "step": 2
    }

    nb_epochs = config["nb_epochs"]
    samples_per_epoch = config["samples_per_epoch"]
    batch_size = config["batch_size"]
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

        if epoch%args.checkpoint_interval == 0:
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
