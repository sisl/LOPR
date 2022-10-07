import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))

from src.prediction.model import PredictionDynamicsEngineLSTM

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