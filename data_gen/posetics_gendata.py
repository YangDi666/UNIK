import argparse
import os
import numpy as np
import json
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm

num_joint = 17
max_frame = 300
num_person_out = 2
num_person_in = 2
coco=['Nose', 'Mouth' ,'Neck' ,'Chest' ,'Mhip' ,'Lsho' ,'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank' ,'Rank']
lcrnet={'Nose': 12, 'Mouth':12, 'Lsho':11 ,'Rsho':10, 'Lelb':9, 'Relb':8, 'Lwri':7, 'Rwri':6, 'Lhip':5, 'Rhip':4, 'Lkne':3, 'Rkne':2, 'Lank':1 ,'Rank':0}
class Feeder_posetics(Dataset):
    '''
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        window_size: The length of the output sequence
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    '''
    

    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 window_size=-1,
                 num_person_in=2,
                 num_person_out=2):
        self.data_path = data_path
        self.label_path = label_path
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()
    
    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        zeros=np.where(X==0)
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        center= X/w*2 - [1, h/w]
        center[zeros]=0
        return center

    def load_data(self):
        # load file list
        self.sample_name = os.listdir(self.data_path)
        
        # load label
        label_path = self.label_path
        with open(label_path) as f:
            label_info = json.load(f)
        self.sample_id =[f[-30:-19] for f in self.sample_name]
        self.label = np.array([label_info[id]['label_index'] for id in self.sample_id])
        has_skeleton = np.array([label_info[id]['has_skeleton'] for id in self.sample_id])

        for k, n in enumerate(self.sample_id):
            if self.sample_name[k].find(n)==-1:
                print(self.sample_name[k])
                print(n)
    
        # ignore the samples which does not has skeleton sequence
        if self.ignore_empty_sample:
            self.sample_name = [s for h, s in zip(has_skeleton, self.sample_name) if h]
            self.label = self.label[has_skeleton]

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  # sample
        self.C = 5 # channel
        self.T = max_frame  # frame
        self.V = num_joint  # joint
        self.M = self.num_person_out  # person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)

        op_name = self.sample_id[index]+'.json'
        with open(sample_path, 'r') as f:
            video_info = json.load(f)
        
        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_out))
        for n, f in enumerate(video_info['frames']):
            if len(f)!=0:
            
                for m, b in enumerate(f):
                    for j, k in enumerate(coco):
                        if m < self.num_person_out:      
                            if k=='Mhip':                               
                                data_numpy[:, n, j, m]=[(b['pose2d'][4]+b['pose2d'][5])/2, (b['pose2d'][17]+b['pose2d'][18])/2, (b['pose3d'][4]+b['pose3d'][5])/2, (b['pose3d'][17]+b['pose3d'][18])/2, (b['pose3d'][30]+b['pose3d'][31])/2]
                            elif k=='Neck':
                                data_numpy[:, n, j, m]=[ (b['pose2d'][10]+b['pose2d'][11])/2, (b['pose2d'][23]+b['pose2d'][24])/2, (b['pose3d'][10]+b['pose3d'][11])/2, (b['pose3d'][23]+b['pose3d'][24])/2, (b['pose3d'][36]+b['pose3d'][37])/2]
                            elif k=='Chest':
                                data_numpy[:, n, j, m]=[((b['pose2d'][4]+b['pose2d'][5])/2+(b['pose2d'][10]+b['pose2d'][11])/2)/2, ((b['pose2d'][17]+b['pose2d'][18])/2+(b['pose2d'][23]+b['pose2d'][24])/2)/2, ((b['pose3d'][4]+b['pose3d'][5])/2+(b['pose3d'][10]+b['pose3d'][11])/2)/2, ((b['pose3d'][17]+b['pose3d'][18])/2+(b['pose3d'][23]+b['pose3d'][24])/2)/2, ((b['pose3d'][30]+b['pose3d'][31])/2+(b['pose3d'][36]+b['pose3d'][37])/2)/2]
                            else:
                                data_numpy[:, n, j, m]=[ b['pose2d'][lcrnet[k]], b['pose2d'][lcrnet[k]+13], b['pose3d'][lcrnet[k]], b['pose3d'][lcrnet[k]+13], b['pose3d'][lcrnet[k]+26]]
                        
                        else:
                            pass
             
                data_numpy[:, n, 1, :] = (data_numpy[:, n, 0, :] + data_numpy[:, n, 2, :])/2 

        # centralization
        data_numpy = data_numpy.transpose(3, 1, 2, 0)
        for i in range(data_numpy.shape[0]):
            keypoints=data_numpy[i,:,:,:2]
            keypoints = self.normalize_screen_coordinates(keypoints[..., :2], w=640, h=480)
            data_numpy[i,:,:,:2]=keypoints
        data_numpy = data_numpy.transpose(3, 1, 2, 0)
        
        # get & check label index
        label=self.label[index]
        
        return data_numpy, label


def gendata(data_path, label_path,
            data_out_path, label_out_path,
            num_person_in=num_person_in,  # observe the first 5 persons
            num_person_out=num_person_out,  # then choose 2 persons with the highest score
            max_frame=max_frame):
    feeder = Feeder_posetics(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame)

    sample_name = feeder.sample_name
    sample_label = []

    fp = np.zeros((len(sample_name), 5, max_frame, num_joint, num_person_out), dtype=np.float32)
    for i, s in enumerate(tqdm(sample_name)):
      
        data, label = feeder[i]
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)
           
    print(len(sample_label),i+1)
    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    np.save(data_out_path, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Posetics-skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='../data/Posetics_raw/')
    parser.add_argument(
        '--out_folder', default='../data/posetics')
    arg = parser.parse_args()

    part = ['train', 'val']
    for p in part:
        print('posetics ', p)
        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        data_path = '{}/posetics_{}'.format(arg.data_path, p)
        label_path = '{}/posetics_{}_label.json'.format('../data/Posetics_raw/', p)
        data_out_path = '{}/{}_data_joint.npy'.format(arg.out_folder, p)
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)

        gendata(data_path, label_path, data_out_path, label_out_path)
