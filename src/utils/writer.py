import torch
import wandb

from torch.utils.tensorboard import SummaryWriter



class Writer():
    def __init__(self, log_dir, name, config):
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        wandb.init(project='Ford', group=f'Dynamics Engine', name=name, dir=log_dir, config=config)

    def add_logs(self, logs, itr, mode):
        for key in logs.keys():
            self.add_scalar(f'{mode}/{key}', logs[key], itr)

    def add_scalar(self, name, value, itr):
        self.tb_writer.add_scalar(name, value, itr)
        wandb.log({name: value, 'step': itr})

    def add_image(self, name, img, itr):
        # self.tb_writer.add_image(name, img, global_step=itr)
        pass


    def close(self):
        self.tb_writer.close()
        wandb.finish()


