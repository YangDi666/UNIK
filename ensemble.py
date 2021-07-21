import argparse
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='smarthome/xsub', help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

dataset = arg.dataset
label = open('data/' + dataset + '/val_label.pkl', 'rb')

label = np.array(pickle.load(label))

r1 = open('./work_dir/' + dataset + '/unik_test_joint/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
print(len(r1))
r2 = open('./work_dir/' + dataset + '/unik_test_bone/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
print(len(r2))
right_num = total_num = right_num_5 = 0
predict=[]
true=[]
if_r = open('./work_dir/' + dataset + '/result_ensemble.txt', 'w')

for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    r = r11 + r22 * arg.alpha
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
    predict.append(r)
    true.append(int(l))
for i, x in enumerate(predict):
    if_r.write(str(x) + ',' + str(true[i]) + '\n')

acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)
