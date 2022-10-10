import torch
import torch.nn as nn

from src.prediction.model import PredictionDynamicsEngineLSTM

l1_loss = nn.L1Loss(reduction='none')

class PredictionLSTM(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        super(PredictionLSTM, self).__init__()

        self.z_c_recon_coeff = config["z_c_recon_coeff"]
        self.z_t_recon_coeff = config["z_t_recon_coeff"]

        self.extrap_t = 5
        self.nt = 20
        self.PredictionDE = PredictionDynamicsEngineLSTM(config["DynamicsEngine"]).cuda()

    def forward(self, z_theme, z_content):
        z_theme_pred, z_content_pred = self.PredictionDE(z_theme[:,:self.extrap_t], z_content[:,:self.extrap_t])
        return z_theme_pred, z_content_pred

def predict(z_theme, z_content, model):
    model.eval()

    with torch.no_grad():
        z_theme_pred, z_content_pred = model(z_theme, z_content)
    model.train()

    return z_theme_pred, z_content_pred

def evaluate_loss(z_theme, z_content, model, logs=None):

    z_theme_pred, z_content_pred = model(z_theme, z_content)


    z_t_recon_loss = l1_loss(z_theme_pred, z_theme[:,1:]).mean(2)
    z_c_recon_loss = l1_loss(z_content_pred, z_content[:,1:]).mean((2,3,4))

    loss = model.z_c_recon_coeff * z_c_recon_loss + \
           model.z_t_recon_coeff * z_t_recon_loss

    loss = loss.mean()

    if logs is None or len(logs.keys()) == 0:
        logs = {
        "Loss/z_c_recon_loss": z_c_recon_loss.mean(),
        "Loss/z_t_recon_loss": z_t_recon_loss.mean(),
        }
    else:
        logs["Loss/z_c_recon_loss"] += z_c_recon_loss.mean()
        logs["Loss/z_t_recon_loss"] += z_t_recon_loss.mean()

    return loss, logs
