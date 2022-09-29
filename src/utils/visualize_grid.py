import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))
import torch

import matplotlib.pyplot as plt

from src.prediction.prediction import evaluate_loss, predict
import numpy as np

def image_grid(array):
    array = array.permute((0, 1, 3, 4, 2))
    array = array.cpu().numpy()

    print(f'Array shape:{array.shape}')

    nrows, ncols, height, width, channels = array.shape
    
    img_grid = (array.reshape(nrows, ncols, height, width, channels)
            .swapaxes(1,2)
            .reshape(height*nrows, width*ncols, channels))
    
    return img_grid

def save_image_grid(path, img):
    img_grid = image_grid(img)
    fig = plt.figure(figsize=(10.,10.))
    plt.imshow(img_grid)
    plt.savefig(f'{path}', dpi=300)
    plt.close(fig)

def visualize_prediction(data_sample_train, data_sample_val, model, path):
    prediction_train = predict(data_sample_train, model)
    prediction_val = predict(data_sample_val, model)

    # prediction_train = image_grid(prediction_train)
    # prediction_val = image_grid(prediction_val)

    save_image_grid(f'{path}_train', prediction_train)
    save_image_grid(f'{path}_val', prediction_val)

    return prediction_train, prediction_val
