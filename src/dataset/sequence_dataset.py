import enum
from re import L
import torch
import torch.utils.data as data
import os  
from PIL import Image
import time
import numpy as np
import pickle
#import hickle as hkl
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))

from src.utils.visualize_grid import image_grid, save_image_grid, visualize_prediction
import json
import cv2

class FasterLatentSequenceDataset(data.Dataset):
    def __init__(self, config, mode, new=False):

        self.datafile = f'{config["datafile"]}/{mode}'
        self.data = f'{config["data"]}/{mode}'
        self.nt = config["nt"]
        self.mode = config["mode"]
        self.new = new

        assert self.mode in ['all', 'unique'], f'Provided an incompatible mode:{self.mode}. Expected all or unique'

        print('Preparing the dataset...')
        self.prep_the_dataset()


    def __getitem__(self, index):
        scene, start_idx = self.possible_starts[index]
        end_idx = start_idx + 2*self.nt #Dataset is in 20Hz
        z_themes, z_contents = [], []
        data = self.load_latents_from_hickles(os.path.join(self.data, str(scene)))
        for loc in range(start_idx, end_idx, 2):
            z_theme, z_content = data[loc]['theme_mu'], data[loc]['spatial_mu']
            z_themes.append(z_theme)
            z_contents.append(z_content)
        
        del data

        z_themes = torch.from_numpy(np.stack(z_themes, 0))
        z_contents = torch.from_numpy(np.stack(z_contents, 0))

        if self.new:
            return (z_themes, z_contents, scene, start_idx) 
        else:
            return (z_themes, z_contents)
        #return (z_themes, z_contents) #TODO: Modify this to include (z_themes, z_contents, scene, start_idx) so we can get a camera feed from nuscenes dataset

    def __len__(self):
        return len(self.possible_starts)

    def prep_the_dataset(self):
        self.scenes = os.listdir(self.datafile)
        self.scene_len = {}
        
        #Measure the length of the scene.
        for scene in self.scenes:
            frames = os.listdir(os.path.join(self.datafile, scene))
            for frame in frames:
                frame = int(frame)
                if scene in self.scene_len.keys():
                    if frame > self.scene_len[scene]:
                        self.scene_len[scene] = frame + 1
                else:
                    self.scene_len[scene] = 1

        # print(f'Indetified scenes and lengths:{self.scene_len}')

        #Identify possible_starts
        possible_starts = []
        cur_loc = 0

        for scene in self.scene_len.keys():
            _len = self.scene_len[scene]


            if self.mode == 'all':
                starts = list(range(0,_len-2*self.nt))
            elif self.mode == 'unique':
                starts = list(range(0,_len-2*self.nt,2*self.nt))

            for idx in starts:
                possible_starts.append((scene,idx))
            self.possible_starts = possible_starts
        print(f'Identified possible starts.')

    def load_latents_from_pickles(self, path):
        infile = open(path, 'rb')
        data = pickle.load(infile)
        infile.close()
        return data['theme_mu'], data['spatial_mu']


    def load_latents_from_hickles(self, path):
        #print(f'Loading')
        data = hkl.load(path)
        return data

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
        #return (z_themes, z_contents) #TODO: Modify this to include (z_themes, z_contents, scene, start_idx) so we can get a camera feed from nuscenes dataset

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
        #return (z_themes, z_contents) #TODO: Modify this to include (z_themes, z_contents, scene, start_idx) so we can get a camera feed from nuscenes dataset

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
            # for frame in frames:
            #     frame = int(frame)
            #     if scene in self.scene_len.keys():
            #         if frame > self.scene_len[scene]:
            #             self.scene_len[scene] = frame + 1
            #     else:
            #         self.scene_len[scene] = 1

        # print(f'Indetified scenes and lengths:{self.scene_len}')

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

class LatentSequenceDatasetWithMaps(data.Dataset):
    def __init__(self, config, mode, new=False):

        self.datafile = f'{config["datafile"]}/{mode}'
        self.datafile_maps = '/home/benksy/Projects/Datasets/Nuscenes_processed_maps'
        self.nt = config["nt"]
        self.mode = config["mode"]
        self.new = new

        assert self.mode in ['all', 'unique'], f'Provided an incompatible mode:{self.mode}. Expected all or unique'

        print('Preparing the dataset...')
        self.prep_the_dataset()


    def __getitem__(self, index):
        scene, start_idx = self.possible_starts[index]
        end_idx = start_idx + 2*self.nt #Dataset is in 20Hz
        z_themes, z_contents = [], []

        for loc in range(start_idx, end_idx, 2):
            z_theme, z_content = self.load_latents_from_pickles(os.path.join(self.datafile, str(scene), str(loc)))
            z_themes.append(z_theme)
            z_contents.append(z_content)

        # print(self.datafile_maps, scene, start_idx)
        scene = int(scene)
        path = self.datafile_maps + f'scene-{scene:0{4}}/' + f'{start_idx + 4*2}_driveable.png'
        # print(path)
        road_layout = cv2.imread(os.path.join(self.datafile_maps, f'scene-{scene:0{4}}', f'{start_idx + 4*2}_driveable.png'))
        ped_cross = cv2.imread(os.path.join(self.datafile_maps, f'scene-{scene:0{4}}', f'{start_idx + 4*2}_ped_cross.png'))
        stop_line = cv2.imread(os.path.join(self.datafile_maps, f'scene-{scene:0{4}}', f'{start_idx + 4*2}_stop_line.png'))

       # print(road_layout.shape)
        maps = np.concatenate([road_layout, ped_cross, stop_line], axis=2)
        maps = np.transpose(maps, axes=(2,0,1))/255 

        z_themes = np.stack(z_themes, 0)
        z_contents = np.stack(z_contents, 0)

        if self.new:
            return (z_themes, z_contents, maps, scene, start_idx) 
        else:
            return (z_themes, z_contents, maps)
        #return (z_themes, z_contents) #TODO: Modify this to include (z_themes, z_contents, scene, start_idx) so we can get a camera feed from nuscenes dataset

    def __len__(self):
        return len(self.possible_starts)

    def prep_the_dataset(self):
        self.scenes = os.listdir(self.datafile)
        self.scene_len = {}
        
        #Measure the length of the scene.
        for scene in self.scenes:
            frames = os.listdir(os.path.join(self.datafile, scene))
            for frame in frames:
                frame = int(frame)
                if scene in self.scene_len.keys():
                    if frame > self.scene_len[scene]:
                        self.scene_len[scene] = frame + 1
                else:
                    self.scene_len[scene] = 1

        # print(f'Indetified scenes and lengths:{self.scene_len}')

        #Identify possible_starts
        possible_starts = []
        cur_loc = 0

        for scene in self.scene_len.keys():
            _len = self.scene_len[scene]

            if self.mode == 'all':
                starts = list(range(0,_len-2*self.nt))
            elif self.mode == 'unique':
                starts = list(range(0,_len-2*self.nt,2*self.nt))

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

class SequenceDatasetDriveGAN(data.Dataset):
    def __init__(self, config, mode):

        self.datafile = f'{config["datafile"]}/{mode}'
        self.nt = config["nt"]
        self.mode = config["mode"]
        self.train = mode #TODO:FIX IT
        self.size = 128
        self.img_transform = None

        assert self.mode in ['all', 'unique'], f'Provided an incompatible mode:{self.mode}. Expected all or unique'

        print('Preparing the dataset...')
        self.prep_the_dataset()


    def __getitem__(self, index):
        scene, start_idx = self.possible_starts[index]
        end_idx = start_idx + 2*self.nt
        frames = []

        for loc in range(start_idx, end_idx, 2):
            frame = self.load_frame(os.path.join(self.datafile, self.scenes[scene][loc][1]))
            #frame = Image.open(os.path.join(self.datafile, self.scenes[scene][loc])) #TODO:Change this #f'{scene}_{loc}.png'
            #print(f'Frame shape:{frame.shape}')
            frames.append(frame)

        frames = np.stack(frames, 0)

        return frames

    def __len__(self):
        return len(self.possible_starts)

    def load_frame(self, fn):
        cur_s = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        cur_s = cv2.cvtColor(cur_s, cv2.COLOR_BGR2RGB)

        cur_s = cur_s / 255.0
        if cur_s.shape[1] != self.size and cur_s.shape[2] != int(self.size*self.width_mul):
            cur_s = cv2.resize(cur_s, (int(self.size*self.width_mul),  self.size))
        s_t = (np.transpose(cur_s, axes=(2, 0, 1))).astype('float32')
        s_t = torch.FloatTensor((s_t - 0.5) / 0.5)
        if self.img_transform is not None:
            s_t = self.img_transform(s_t)

        return s_t


    def prep_the_dataset(self):
        #self.files = os.listdir(self.datafile)
        json_f = open(f'/home/benksy/Projects/Datasets/NuscenesImgDatasetWithIdsStyleGAN/{self.train}/scene_ids.json')
        self.files = json.load(json_f)
        self.scenes_len = {}
        self.scenes = {}

        #Measure the length of the scene.
        for f in self.files.keys():
            scene, frame = self.files[f].split('_')
            frame = int(frame[:-4])
            scene = int(scene)

            if scene not in self.scenes.keys():
                self.scenes[scene] = [(frame,f)]
                self.scenes_len[scene] = 1
            else:
                self.scenes[scene].append((frame,f))
                self.scenes_len[scene] += 1

        #print(f'Indetified scenes and lengths:{self.scenes_len}')

        #Identify possible_starts
        possible_starts = []
        cur_loc = 0

        def myFunc(e):
            return e[0]

        for scene in self.scenes_len.keys():
            _len = self.scenes_len[scene]
            self.scenes[scene].sort(key=myFunc)

            if self.mode == 'all':
                starts = list(range(0,_len-2*self.nt))
            elif self.mode == 'unique':
                starts = list(range(0,_len-2*self.nt,2*self.nt))

            for idx in starts:
                possible_starts.append((scene,idx))
            self.possible_starts = possible_starts
        print(f'Identified possible starts.')
        json_f.close()

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


if __name__ == "__main__":
    config = {
        "datafile": "/home/benksy/Projects/Datasets/NuscenesImgDatasetWithIdsStyleGAN",
        "nt": 20,
        "mode": "all"
    }

    dataset = SequenceDataset(config, 'train', freq=20)
    
    work = 0
    B = 1
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size = B, shuffle = False, num_workers=work)
    out, scenes, frames = dataset[120]#train_dataloader[90]
    print(f'Shape:{out.shape}. Scenes:{scenes}. Frames:{frames}')

    save_image_grid('example', torch.from_numpy(out).unsqueeze(0))
