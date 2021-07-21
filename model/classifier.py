import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .backbone_unik import *

class Model(nn.Module):
    def __init__(self, num_class=60, num_joints=25, num_person=2, tau=1, num_heads=3, in_channels=2, drop_out=0, backbone_fixed=False, weights='no' ):
        super(Model, self).__init__()
        self.backbone_fixed = backbone_fixed
        
         #load model
        self.model_action = UNIK(320, num_joints, num_person, tau, num_heads, in_channels=in_channels)

        if weights!='no':
            print('pre-training: ', weights)
            weights = torch.load(weights)
            weights = OrderedDict(
                [[k.split('module.')[-1],
                v.cuda(0)] for k, v in weights.items()])

            keys = list(weights.keys())
            try:
                self.model_action.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
        

        # transfer learning
        self.model_action.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.model_action.fc.weight, 0, math.sqrt(2. / num_class))
        #print('parameters: ', len(list(self.model_action.fc.parameters())))
        
        # freeze the backbone for Linear evaluation
        if self.backbone_fixed:
            for l, module in self.model_action._modules.items():
                if l !='fc':
                    print('fixed layers:', l)
                    for p in module.parameters():
                        p.requires_grad=False

    def get_pad(self):
        return int(self.pad)

    def forward(self, x):
        x = self.model_action(x)
        return  x


