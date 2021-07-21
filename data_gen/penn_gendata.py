import argparse
import pickle
from tqdm import tqdm
import sys
import json
import pandas as pd

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization2d

# joints distrubution
joints = ['head', 'nose' ,'Neck' ,'Chest' ,'Mhip' ,'Lsho' ,'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank' ,'Rank']
lcrnet = {'nose': 12, 'head':12, 'Lsho':11, 'Rsho':10, 'Lelb':9, 'Relb':8, 'Lwri':7, 'Rwri':6, 'Lhip':5, 'Rhip':4, 'Lkne':3, 'Rkne':2, 'Lank':1 ,'Rank':0}


max_body_true = 2
max_body=2
num_joint = 17
max_frame = 700

import numpy as np
import os


def read_skeleton_filter(file):
    with open(file, 'r') as json_data:
        skeleton_sequence = json.load(json_data)

    return skeleton_sequence


def get_nonzero_std(s):  
    index = s.sum(-1).sum(-1) != 0  
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() 
    else:
        s = 0
    return s


def normalize_screen_coordinates( X, w, h):
    assert X.shape[-1] == 2
    zeros=np.where(X==0)
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    center= X/w*2 - [1, h/w]
    center[zeros]=0
    return center

def read_xy(file, max_body=2, num_joint=17): 
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, len(seq_info['frames']), num_joint, 2))
    for n, f in enumerate(seq_info['frames']):
        if len(f)!=0:
            for m, b in enumerate(f):
                for j,k in enumerate(joints):
                    if m < max_body:
                        if k=='Mhip':
                            data[m, n, j, :]=[ (b['pose2d'][4]+b['pose2d'][5])/2, (b['pose2d'][17]+b['pose2d'][18])/2 ]
                        elif k=='Neck':
                            data[m, n, j, :]=[ (b['pose2d'][10]+b['pose2d'][11])/2, (b['pose2d'][23]+b['pose2d'][24])/2 ]
                        elif k=='Chest':
                            data[m, n, j, :]=[ ((b['pose2d'][4]+b['pose2d'][5])/2+(b['pose2d'][10]+b['pose2d'][11])/2)/2, ((b['pose2d'][17]+b['pose2d'][18])/2+(b['pose2d'][23]+b['pose2d'][24])/2)/2 ]
                        else:
                            data[m, n, j, :]=[ b['pose2d'][lcrnet[k]], b['pose2d'][lcrnet[k]+13] ]
                    else:
                        pass
   
    # centralization
    for i in range(data.shape[0]):
        keypoints=data[i,:,:,:2]
        #print('kpts: ', keypoints.shape) 512*424 depth
        keypoints = normalize_screen_coordinates(keypoints[..., :2], w=640, h=480)
        data[i,:,:,:2]=keypoints    
   
    data = data.transpose(3, 1, 2, 0)
    data[:,:,1,:] = (data[:,:,0,:] + data[:,:,2,:]) / 2
    return data


def gendata(data_path, annot_path, out_path, ignored_sample_path=None,  part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.json' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    annot=pd.read_csv(annot_path)
    for filename in os.listdir(data_path):
        if (filename in ignored_samples) or filename[0]=='S' or int(filename[:-5])>=2326:
            continue
   
        action_class=int(annot[annot['video']==int(filename[:-5])]['category_code'])
        istrain=int( annot[annot['video']==int(filename[:-5])]['train']  )+1
        
        if part=='train':
            issample=istrain
        else:
            issample=not istrain

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 2, max_frame, num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = read_xy(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
    
    fp[:,:,:,:,:] = pre_normalization2d(fp[:,:,:,:,:])

    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Penn Data Converter.')
    parser.add_argument('--data_path', default='../data/pennAction_raw/skeletons/')
    parser.add_argument('--ignored_sample_path',
                        default=None)
    parser.add_argument('--out_folder', default='../data/penn/')

   
    part = ['train','val']
    arg = parser.parse_args()
    print('skeleton path: ', arg.data_path)
    annot_path='../data/pennAction_raw/pennaction_complete.csv'
    for p in part:
        out_path = arg.out_folder
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print( p)
        gendata(
                arg.data_path,
                annot_path,
                out_path,
                arg.ignored_sample_path,
                part=p)
