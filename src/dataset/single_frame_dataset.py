from torch.utils.data import Dataset
import os
import random
import numpy as np
import cv2
import pickle
from src.representation_learning.model.operations import check_arg
import torch
import json

def list_to_dict(l):
    d = {}
    for entry in l:
        d[entry] = 1
    return d

class RGBImageDataset(Dataset):
    def __init__(self, path, dataset, size, args=None, train=True):
        datadir = path
        paths = []
        self.size = size
        self.dataset = dataset
        self.args = args
        self.width_mul = 1
        self.img_transform = None
        self.is_pilotnet = False
        self.dataset = dataset
        paths = []
        for scene in os.listdir(path):
            path_to_scene = f'{path}/{scene}'
            if path_to_scene[-4:] == 'json':
                print(path_to_scene)
                continue
            for sensor in os.listdir(path_to_scene):
                path_to_sensor = f'{path_to_scene}/{sensor}'
                for image_path in os.listdir(path_to_sensor):
                    key = f'{path_to_sensor}/{image_path}'
                    paths.append(key)

        print(self.dataset + ', Number of data: ' + str(len(paths)))
        random.Random(4).shuffle(paths)

        self.samples = paths
        # f = open(f'{path}/scene_ids.json')
        # self.scene_ids = json.load(f)
        # f.close()


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        fn = self.samples[index]   
        cur_s = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        cur_s = cv2.cvtColor(cur_s, cv2.COLOR_BGR2RGB)

        cur_s = cur_s / 255.0

        if cur_s.shape[1] != self.size and cur_s.shape[2] != self.size:
            print('resizing!')
            cur_s = cv2.resize(cur_s, (self.size,  self.size))
        s_t = (np.transpose(cur_s, axes=(2, 0, 1))).astype('float32')
        s_t = torch.FloatTensor((s_t - 0.5) / 0.5)
        
        return s_t #, self.scene_ids[f'{folder}/{frame}']

class OGMImageDataset(Dataset):
    def __init__(self, path, dataset, size, args=None, train=True, with_scene_info=False):
        datadir = path
        paths = []
        self.size = size
        self.dataset = dataset
        self.args = args
        self.dataset = 'nuscenes_og'
        self.with_scene_info = with_scene_info

        paths = []
        for folder in os.listdir(path):
            path_to_folder = f'{path}/{folder}'
            if path_to_folder[-4:] == 'json':
                print(path_to_folder)
                continue
            for frame in os.listdir(path_to_folder):
                key = f'{path_to_folder}/{frame}'
                paths.append(key)
                #print(key)

        print(self.dataset + ', Number of data: ' + str(len(paths)))
        random.Random(4).shuffle(paths)

        self.samples = paths

        if self.with_scene_info:
            f = open(f'{path}/scene_ids.json')
            self.scene_ids = json.load(f)
            f.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fn = self.samples[index]

        cur_s = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        cur_s = cv2.cvtColor(cur_s, cv2.COLOR_BGR2RGB)
        cur_s = cur_s / 255.0

        s_t = (np.transpose(cur_s, axes=(2, 0, 1))).astype('float32')
        s_t = torch.FloatTensor((s_t - 0.5) / 0.5)

        # except:
        #     print(fn)
        #     return None

        if self.with_scene_info:
            folder, frame = str(fn).split('/')[-2:]
            return s_t, self.scene_ids[f'{folder}/{frame}']
        else:
            return s_t