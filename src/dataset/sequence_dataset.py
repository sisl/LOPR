import enum
from re import L
import torch
import torch.utils.data as data
import os  
from PIL import Image
import time
import numpy as np
import pickle
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))

from src.utils.visualize_grid import image_grid, save_image_grid, visualize_prediction
import json
import cv2

class LatentSequenceDataset(data.Dataset):
    def __init__(self, config, mode, new=False):

        self.datafile = f'{config["datafile"]}/{mode}'
        self.nt = config["nt"]
        self.mode = config["mode"]
        self.step = config["step"]
        self.new = new

        assert self.mode in ['all', 'unique'], f'Provided an incompatible mode:{self.mode}. Expected all or unique'

        print('Preparing the dataset...')
        self.prep_the_dataset()


    def __getitem__(self, index):
        scene, start_idx = self.possible_starts[index]
        end_idx = start_idx + self.step*self.nt #Dataset is in 20Hz
        z_themes, z_contents = [], []

        for loc in range(start_idx, end_idx, self.step):
            z_theme, z_content = self.load_latents_from_pickles(os.path.join(self.datafile, str(scene), str(loc)))
            z_themes.append(z_theme)
            z_contents.append(z_content)

        z_themes = np.stack(z_themes, 0)
        z_contents = np.stack(z_contents, 0)

        if self.new:
            return (z_themes, z_contents, scene, start_idx) 
        else:
            return (z_themes, z_contents)
       
    def get(self, scene, start_idx):
        end_idx = start_idx + self.step*self.nt #Dataset is in 20Hz
        z_themes, z_contents = [], []

        for loc in range(start_idx, end_idx, self.step):
            z_theme, z_content = self.load_latents_from_pickles(os.path.join(self.datafile, str(scene), str(loc)))
            z_themes.append(z_theme)
            z_contents.append(z_content)

        z_themes = np.stack(z_themes, 0)
        z_contents = np.stack(z_contents, 0)

        if self.new:
            return (z_themes, z_contents, scene, start_idx) 
        else:
            return (z_themes, z_contents)

    def __len__(self):
        return len(self.possible_starts)

    def prep_the_dataset(self):
        self.scenes = os.listdir(self.datafile)
        self.scene_len = {}
        
        print(f'MODE:{self.mode} Datafile:{self.datafile}')
        
        #Measure the length of the scene.
        for scene in self.scenes:
            frames = os.listdir(os.path.join(self.datafile, scene))
            self.scene_len[scene] = len(frames)

        #Identify possible_starts
        possible_starts = []
        cur_loc = 0

        for scene in self.scene_len.keys():
            _len = self.scene_len[scene]

            if self.mode == 'all':
                starts = list(range(0,_len*self.step-self.nt*self.step, self.step))
            elif self.mode == 'unique':
                starts = list(range(0,_len*self.step-self.nt*self.step,self.step*self.nt))

            for idx in starts:
                possible_starts.append((scene,idx))
            self.possible_starts = possible_starts
        # print(f'Identified possible starts.')

    def load_latents_from_pickles(self, path):
        infile = open(path, 'rb')
        data = pickle.load(infile)
        infile.close()
        return data['theme_mu'], data['spatial_mu']


    def load_latents_from_hickles(self, path):
        data = hkl.load(path)
        return data['theme_mu'], data['spatial_mu']

class SequenceDataset(data.Dataset):
    def __init__(self, config, mode, freq=10):

        self.datafile = f'{config["datafile"]}/{mode}'
        self.nt = config["nt"]
        self.mode = config["mode"]
        if freq == 10:
            self.step = 2
        else:
            self.step = 1

        assert self.mode in ['all', 'unique'], f'Provided an incompatible mode:{self.mode}. Expected all or unique'

        print('Preparing the dataset...')
        self.prep_the_dataset()

    def __getitem__(self, index):
        scene, start_idx = self.possible_starts[index]
        print(f'Data:{scene}. Start_idx:{start_idx}')
        end_idx = start_idx + self.step*self.nt
        frames = []

        for loc in range(start_idx, end_idx, self.step):
            #print(f'Loading:{self.scenes[scene][loc]}')
            frame = Image.open(os.path.join(self.datafile, self.scenes[scene][loc])) #TODO:Change this #f'{scene}_{loc}.png'
            frames.append(frame)

        frames = np.stack(frames, 0)
        frames = frames[:,np.newaxis,:,:]

        frames = frames / 127.5 - 1

        return frames, scene, start_idx

    def __len__(self):
        return len(self.possible_starts)

    def prep_the_dataset(self):
        json_f = open('/home/benksy/Projects/Datasets/NuscenesImgDatasetWithIdsStyleGAN/train/scene_ids.json')
        self.files = json.load(json_f)
        self.scenes_len = {}
        self.scenes = {}

        #Measure the length of the scene.
        for f in self.files.keys():
            scene, frame = self.files[f].split('_')
            frame = int(frame[:-4])
            scene = int(scene)

            if scene not in self.scenes.keys():
                self.scenes[scene] = {frame: f}
                self.scenes_len[scene] = 1
            else:
                self.scenes[scene][frame] = f
                self.scenes_len[scene] += 1

        #Identify possible_starts
        possible_starts = []
        cur_loc = 0

        for scene in self.scenes_len.keys():
            _len = self.scenes_len[scene]

            if self.mode == 'all':
                starts = list(range(0,_len-2*self.nt))
            elif self.mode == 'unique':
                starts = list(range(0,_len-2*self.nt,2*self.nt))

            for idx in starts:
                possible_starts.append((scene,idx))
            self.possible_starts = possible_starts
        json_f.close()

