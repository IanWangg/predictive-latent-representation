import torch
from torch.utils import data
import gym
import sys
sys.path.append('../d4rl-atari')
import numpy as np
import d4rl_atari


# the resulting sequences are overlapping
class Atari(data.Dataset):
    def __init__(self,
                 transform = None,
                 seq_len = 4,
                 num_seq = 6,
                 downsample = 3,
                 epsilon = 5,
                 dataset = None,
                 stack = True,
                 n_channels = 5,
                 return_actions = False,
                 overlapping = True):
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon 

        # if the returned frames are not overlapping, we must return actions
        # assert (not overlapping and return_actions) or overlapping
        self.return_actions = False
        self.overlapping = True
        self.return_actions = return_actions

        # create the env
        assert dataset is not None

        # get the dataset
        self.dataset = dataset
        
        self.observations = dataset['observations']
        self.terminals = dataset['terminals']
        self.actions = dataset['actions']
        
        traj_start = [0]
        self.terminals = self.dataset['terminals']
        for i, done in enumerate(self.terminals):
            if done and i != len(self.terminals) - 1:
                traj_start.append(i + 1)


        print(f'total trajactories : {len(traj_start)}')
        self.starting_point = traj_start
        self.num_frames = len(self.terminals)
        
        # each time we pick a traj, and sample 8 seqs from it
        
    def __len__(self):
        # drop the last traj
        return len(self.starting_point) - 1
    

    # the behaviour of the idx_sampler decides if the frames are overlapping or not
    def idx_sampler(self, start, end):
        # return shape (num_seq, seq_len, H, W)
        
        # how many frames needed?
        '''
        chang here if we want to change overlapping/non-overlapping

        Note that each frames in the dataset is of (4, 84, 84), which means they are already stacked
        '''
        if self.overlapping:
            frames_needed = self.num_seq
            #print(f'what we need is : num_seq == {self.num_seq} and seq_len = {self.seq_len}')
            seq_start = np.random.choice(range(start, end - frames_needed))
            frames = torch.Tensor(self.observations[seq_start:seq_start+self.num_seq])
            (num_seq, seq_len, H, W) = frames.size()
            #print(num_seq, seq_len)
            frames = frames.contiguous().view(num_seq, 1, seq_len, H, W)
            actions = torch.Tensor(self.actions[seq_start:seq_start+self.num_seq])
            #print(f'what we get is : shape = {frames.shape}')

        else :
            frames_needed = (self.num_seq - 1) * self.seq_len + 1
            seq_start = np.random.choice(range(start, end-frames_needed))
            # pick non-overlapped frames
            frames = torch.Tensor(self.observations[seq_start:seq_start+self.num_seq:self.seq_len])
            actions = torch.Tensor(self.actions[seq_start:seq_start+self.num_seq:self.seq_len])
            (num_seq, seq_len, H, W) = frames.size()
            frames = frames.contiguous().view(num_seq, 1, seq_len, H, W)

            print(frames.shape)
            print(actions.shape)


        if self.return_actions:
            return frames, actions

        else:
            return frames
    
    def __getitem__(self, index):
        # eligible start frame of the first seq
        start = self.starting_point[index]
        
        # eligible end frame of the last seq
        end = self.starting_point[index + 1] - 1
        
        return self.idx_sampler(start, end)    