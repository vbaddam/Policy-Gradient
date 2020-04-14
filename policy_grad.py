import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ACagent(nn.Module):

    def __init__(self):
        super(ACagent, self).__init__()

        self.conv1 = nn.conv2D(1, 16, kernal_size=4, stride= 2)
        self.conv2 = nn.conv2D(16, 32, kernal_size= 2, stride = 1)
        self.fc1 = nn.Linear(2050, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, state):

        [s1, s2, s3] = state

        x1 = F.relu(self.conv1(s1))
        x2 = F.relu(self.conv1(s2))
        
        x1 = F.relu(self.conv2(x1))
        x2 = F.relu(self.conv2(x2))

        x1 = x1.view(-1, 1*32*32)
        x2 = x2.view(-1, 1*32*32)
        x3 = s3.view(-1, 1*2*1)

        x = [x1, x2, x3]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return(x)

    def act(self, state):
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])


        return(highest_prob_action, log_prob)





