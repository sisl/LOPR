import torch
import torch.nn as nn

l1_loss = nn.L1Loss(reduction='none')

def predict(z_theme, z_content, model, maps=None):
    model.eval()

    with torch.no_grad():
        if maps != None:
            z_theme_pred, z_content_pred = model(z_theme, z_content, maps)
        else:
            z_theme_pred, z_content_pred = model(z_theme, z_content)
    model.train()

    return z_theme_pred, z_content_pred

def evaluate_loss(z_theme, z_content, model, logs=None, maps=None):

    if maps != None:
        z_theme_pred, z_content_pred = model(z_theme, z_content, maps)
    else:
        z_theme_pred, z_content_pred = model(z_theme, z_content)


    z_t_recon_loss = l1_loss(z_theme_pred, z_theme[:,1:]).mean(2)
    z_c_recon_loss = l1_loss(z_content_pred, z_content[:,1:]).mean((2,3,4))
    # z_c_recon_loss = l1_loss(z_theme_pred[:,1:], z_theme[:,1:]).mean(2)
    # z_t_recon_loss = l1_loss(z_content_pred[:,1:], z_content[:,1:]).mean((2,3,4))
    # z_c_recon_loss = l1_loss(z_theme_pred, z_theme[:,1:]).mean(2)
    # z_t_recon_loss = l1_loss(z_content_pred, z_content[:,1:]).mean((2,3,4))
    # z_c_recon_loss = l1_loss(z_theme_pred[:,4:], z_theme[:,5:]).mean(2) #check
    # z_t_recon_loss = l1_loss(z_content_pred[:,4:], z_content[:,5:]).mean((2,3,4))

    # print(f'Loss:{z_t_recon_loss.mean(), z_c_recon_loss.mean()}')

    # print(f'Loss img recon:{img_recon_loss.shape}, z_c_recon_loss:{z_c_recon_loss.shape}. z_t_recon_loss:{z_t_recon_loss.shape}')
    # print(f'Cont...shapes :{z_t_kl_loss.shape}, {z_c_kl_loss.shape}')

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

def evaluate_recurrent_loss(z_theme, z_content, model, logs=None):

    z_theme_pred, z_content_pred = model.predict(z_theme, z_content)

    z_c_recon_loss = l1_loss(z_theme_pred, z_theme[:,5:]).mean(2) #check
    z_t_recon_loss = l1_loss(z_content_pred, z_content[:,5:]).mean((2,3,4))

    # print(f'Loss:{z_t_recon_loss.mean(), z_c_recon_loss.mean()}')

    # print(f'Loss img recon:{img_recon_loss.shape}, z_c_recon_loss:{z_c_recon_loss.shape}. z_t_recon_loss:{z_t_recon_loss.shape}')
    # print(f'Cont...shapes :{z_t_kl_loss.shape}, {z_c_kl_loss.shape}')

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
