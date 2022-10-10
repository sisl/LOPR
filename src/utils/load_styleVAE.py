import torch
from src.representation_learning.model.model import styleVAEGAN


def load_styleVAE(config):
    ckpt = torch.load(config["ckpt"])

    #VAE vs VAE_ema: which one to pick?

    vae_model = styleVAEGAN
    vae = vae_model(
        config["size"], config["n_mlp"], channel_multiplier=config["channel_multiplier"], args=config["args"],
    ).to(config["device"])

    vae.load_state_dict(ckpt['vae_ema'])
    del ckpt

    #test_the_styleVAE()

    return vae #_ema

def test_the_styleVAE():
    """
    Print the stats.
    """
    pass
