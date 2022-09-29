import torch
import torch.nn as nn

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))

# from src.prediction.encoder_generator import EncoderGeneratorStyleGAN2
from src.prediction.deterministic_dynamics_engine_lstm import PredictionDynamicsEngineLSTM
from src.prediction.deterministic_dynamics_engine_lstm_with_maps import PredictionDynamicsEngineLSTMwithMaps
from src.prediction.dynamics_engine_transformer import PredictionDynamicsEngineTransformer
from src.prediction.variational_module import VariationalModule
from src.utils.other_utils import load_config

class PredictionTransformer(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        super(PredictionTransformer, self).__init__()

        self.z_c_recon_coeff = config["z_c_recon_coeff"]
        self.z_t_recon_coeff = config["z_t_recon_coeff"]

        self.extrap_t = 5
        self.nt = 20
        self.PredictionDE = PredictionDynamicsEngineTransformer(config["DynamicsEngine"]).cuda()

    def forward(self, z_theme, z_content):
        z_theme_pred, z_content_pred = self.PredictionDE(z_theme, z_content)
        return z_theme_pred, z_content_pred

    def predict(self, z_theme, z_content):
        z_theme_pred, z_content_pred = self.PredictionDE.predict(z_theme[:,:self.extrap_t], z_content[:, :self.extrap_t])
        return z_theme_pred, z_content_pred

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

class PredictionLSTMMapswithMaps(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        super(PredictionLSTMMapswithMaps, self).__init__()

        self.z_c_recon_coeff = config["z_c_recon_coeff"]
        self.z_t_recon_coeff = config["z_t_recon_coeff"]

        self.extrap_t = 5
        self.nt = 20
        self.PredictionDE = PredictionDynamicsEngineLSTMwithMaps(config["DynamicsEngine"]).cuda()

    def forward(self, z_theme, z_content, maps):
        z_theme_pred, z_content_pred = self.PredictionDE(z_theme[:,:self.extrap_t], z_content[:,:self.extrap_t], maps)
        return z_theme_pred, z_content_pred

if __name__ == "__main__":
        path = "/home/benksy/Projects/VAE-Two-latents-StyleGAN/config.yaml"
        config = {
            "img_recon_coeff": 5,
            "z_c_recon_coeff": 0.1,
            "z_t_recon_coeff": 0.1,
            "z_c_kl_coeff": 0.1,
            "z_t_kl_coeff": 0.1
        }

        config["E&G"] = load_config(path)

        config["DynamicsEngine"] = {
            "extrap_t": 5,
            "nt": 20,
            "model": {
                "fc_model": [
                    ['Linear', {'in': 512, 'out': 512}],
                    ['Activation', 'leaky_relu']
                ],
                "conv_model": [
                    ['Conv2D_BN', {'in': 512, 'out': 512, 'bias': True, 'kernel': 1, 'stride': 1, 'activation': 'leaky_relu'}], #(512,4,4)
                    ['AvgPool2d', {'in': 512, 'out': 512, 'kernel': 4}], #O: 512
                    ['Linear', {'in':512, 'out': 512}]
                ],
                "recurrent_model": [
                    ["LSTM", {"input_size": 1024, "hidden_size": 1024, "num_layers": 1, "batch_first": True}]
                ]
            }

        }

        model = Prediction(config)

        grids = torch.randn(1, 20, 1, 128, 128).cuda()
        model(grids)
        # grids, z_theme_pred, z_content_pred, z_theme, z_content, z_t_kl_loss, z_c_kl_loss = model(grids)
        # print(f'Grids shape:{grids.shape}, z_theme:{z_theme_pred.shape, z_theme.shape}, z_content_pred:{z_content_pred.shape, z_content.shape}, KL:{z_t_kl_loss.shape, z_c_kl_loss.shape}')