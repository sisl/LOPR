import torch
import torch.nn as nn

class PredictionDynamicsEngineLSTM(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        super(PredictionDynamicsEngineLSTM, self).__init__()

        self.extrap_t = config["extrap_t"]
        self.nt = config["nt"]

        model_config = config["model"]

        self.fc_model = nn.Sequential()

        idx = 0
        for layer, layer_config in model_config["fc_model"]:
            if layer == "Linear":
                self.fc_model.add_module(f'{layer}_{idx}', nn.Linear(layer_config["in"], layer_config["out"]))
            elif layer == "Activation":
                self.fc_model.add_module(f'{layer}_{idx - 1}', get_activation_fn(layer_config))
            else:
                print(f'Error. {layer} is not implemented.')
            idx += 1    

        self.conv_model = nn.Sequential() 
        self.conv_linear_model = nn.Sequential()

        idx = 0
        for layer, layer_config in model_config["conv_model"]:
            if layer == "Conv2D_BN":
                self.conv_model.add_module(f'{layer}_{idx}', Conv2D_BN(layer_config))
            elif layer == "Linear":
                self.conv_linear_model.add_module(f'{layer}_{idx}', nn.Linear(layer_config["in"], layer_config["out"]))
            elif layer == "AvgPool2d":
                self.conv_model.add_module(f'{layer}_{idx}', nn.AvgPool2d(layer_config["kernel"], stride=None))
            else:
                print(f'Error. {layer} is not implemented.')   
            idx += 1

        layer_config = model_config["recurrent_model"][0][1]

        self.recurrent_model = nn.LSTM(**layer_config)

        # ## Setup h_0, c_0 as a trainable param
        num_layers = model_config["recurrent_model"][0][1]["num_layers"]
        input_size = model_config["recurrent_model"][0][1]["input_size"]
        hidden_cell = model_config["recurrent_model"][0][1]["hidden_size"]
        self.split = model_config["split"]
        
        self.h_0 = nn.Parameter(torch.zeros(num_layers,input_size)) #.cuda()
        self.c_0 = nn.Parameter(torch.zeros(num_layers,hidden_cell)) #.cuda()

        upsample_input_shape = hidden_cell - self.split

        self.conv_output = nn.Sequential()
        if "conv_content_model" in model_config.keys():
            idx = 0
            for layer, layer_config in model_config["conv_content_model"]:
                if layer == "Conv2D_BN":
                    self.conv_output.add_module(f'{layer}_{idx}', Conv2D_BN(layer_config))
                else:
                    print(f'Error. {layer} is not implemented.')   
                idx += 1
            self.upsample_content = nn.ConvTranspose2d(upsample_input_shape, upsample_input_shape, 4)
        else:
            self.conv_output.add_module(f'Identity', nn.Identity())
            self.upsample_content = nn.ConvTranspose2d(upsample_input_shape, 64, 4)

        self.upsample_theme = nn.Linear(128, 128)

    def forward(self, z_theme, z_content):

        assert len(z_content.shape) == 5, f'Expected z_content shape to be [B,T,D, H, W]. Instead, it is:{z_content.shape}'

        B, T, D_content, W, H = z_content.shape
        B, T, D_theme = z_theme.shape

        z_content = z_content.reshape(B*T,D_content, W, H)
        z_content = self.conv_model(z_content)
        z_content = z_content.squeeze(-1)
        z_content = z_content.squeeze(-1)
        z_content = self.conv_linear_model(z_content)
        z_theme = self.fc_model(z_theme)

        z_content = z_content.reshape(B, T, -1)
        z_obs = torch.cat([z_content, z_theme], axis=-1) #(B, T, D=1024)

        #Processing observations.
        h_0 = self.h_0.unsqueeze(1)
        h_0 = h_0.repeat(1,B,1)

        c_0 = self.c_0.unsqueeze(1)
        c_0 = c_0.repeat(1,B,1)

        z_past, (h_n, c_n) = self.recurrent_model(z_obs, (h_0, c_0)) #(B,T,Dout), (2,B,H)
        z_n = z_past[:,-1].unsqueeze(1) #(B,Dout)
    
        prediction = []

        for i in range(self.nt-self.extrap_t-1):
            z_n, (h_n, c_n) = self.recurrent_model(z_n, (h_n, c_n)) 
            prediction.append(z_n) #(B,1,Dout)

        prediction = torch.cat(prediction, 1)

        z = torch.cat([z_past, prediction], 1) #(B,T,Dout)
        output_z_theme, output_z_content = z[:,:,:self.split], z[:,:,self.split:]
        output_z_content = output_z_content.reshape(B*(self.nt-1),-1).unsqueeze(-1).unsqueeze(-1)
        output_z_content = self.upsample_content(output_z_content)
        output_z_content = self.conv_output(output_z_content)
        output_z_content = output_z_content.reshape(B,self.nt-1,64,4,4)
        output_z_theme = self.upsample_theme(output_z_theme)
        
        return output_z_theme, output_z_content

def get_activation_fn(activation):
    #Credits: Francesco Zuppichini

    activations_options = nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['tanh', nn.Tanh()],
        ['none', nn.Identity()]
    ])
    return  activations_options[activation]

class Conv2D_BN(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        super(Conv2D_BN, self).__init__()

        in_channels = config["in"]
        out_channels = config["out"]
        kernel = config["kernel"]
        stride = config["stride"]
        bias = config["bias"]
        activation = config["activation"]
        if "padding" not in config.keys():
            padding = kernel//2
        else:
            padding = config["padding"]

        self.conv_2d_bn = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels, kernel, bias=bias, padding=padding, stride=stride),
                                    nn.BatchNorm2d(out_channels),
                                    get_activation_fn(activation)
                                    )
    def forward(self, x):
        output = self.conv_2d_bn(x)
        return output