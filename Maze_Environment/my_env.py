# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 15:08:24 2020

@author: abdel
"""
from collections import deque
import numpy as np
from PIL import Image
import cv2
import torch
import time

###############################################################################

# The class Args is used only to test the environment afterwards

###############################################################################

class Args():
    def __init__(self,hashtable):
        
        self.seed = hashtable['seed']
        self.target_update = hashtable['target_update']
        self.game = hashtable['game']
        self.T_max = hashtable['T_max']
        self.learn_start = hashtable['learn_start']
        self.memory_capacity = hashtable['memory_capacity']
        self.replay_frequency = hashtable['replay_frequency']
        self.multi_step = hashtable['multi_step']
        self.architecture = hashtable['architecture']
        self.hidden_size = hashtable['hidden_size']
        self.learning_rate = hashtable['learning_rate']
        self.evaluation_interval = hashtable['evaluation_interval']

        self.id = 'rainbow'
        self.max_episode_length = int(108e3)
        self.history_length = 4
        self.noisy_std = 0.1
        self.atoms = 51
        self.V_min = -10
        self.V_max = 10
        self.priority_exponent = 0.5
        self.priority_weight = 0.4
        self.discount = 0.99
        self.reward_clip = 0
        self.adam_eps = 1.5e-4
        self.batch_size = 32
        self.evaluation_episodes = 10
        self.evaluation_size = 500
        self.checkpoint_interval = 0

        self.disable_cuda = False
        self.model = None
        self.evaluate = False
        self.render = False
        self.enable_cudnn = False
        self.memory = None
        self.disable_bzip_memory = False

       
hashtable = dict()

hashtable['seed'] = 123
hashtable['target_update'] = 500
hashtable['game'] = 'hero'
hashtable['T_max'] = 5000
hashtable['learn_start'] = 1600
hashtable['memory_capacity'] = 500000
hashtable['replay_frequency'] = 1
hashtable['multi_step'] = 20
hashtable['architecture'] = 'data-efficient'
hashtable['hidden_size'] = 256
hashtable['learning_rate'] = 0.0001
hashtable['evaluation_interval'] = 500

args = Args(hashtable)

if torch.cuda.is_available() and not args.disable_cuda:
    print("3ndk gpu")
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.enabled = args.enable_cudnn
else:
    args.device = torch.device('cpu')

###############################################################################

# The environment class

###############################################################################


class Env():
    def __init__(self,size,ntraps,args,traps=None,goal=None,start=None):    
        self.size = size
        self.states = np.arange(1,size**2)
        self.ntraps = ntraps
        self.tot_reward = 0
        
        self.m_penalty = -1
        self.o_penalty = -5
        self.t_penalty = -10
        self.g_reward = 10
        
        self.device = args.device
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode
        #There are 4 actions
        self.actions = dict([i, e] for i, e in zip(range(4), range(4)))
        
        if goal == None:
            goal_x = np.random.randint(1,self.size) 
            goal_y = np.random.randint(1,self.size)
            self.goal = (goal_x,goal_y)
        else: 
            self.goal = (goal[0],goal[1])
        
        self.traps = []
        
        dropped = []
        for i in range(-1,2):
            for j in range(-1,2):
                dropped.append((self.goal[0] + i -1)*self.size + self.goal[1] + j)
        
        count = 1
        while count <= ntraps:
            if traps == None:
                x = np.random.randint(1,self.size) 
                y = np.random.randint(1,self.size)
            else:
                x = traps[count-1][0]
                y = traps[count-1][1]
            
            if (x-1)*size+y not in dropped:
                count += 1
                
                self.traps.append((x,y))
                
                for i in range(-1,2):
                    for j in range(-1,2):
                        dropped.append((x + i - 1)*self.size + y + j)
                        
        cond = True
        while cond:
            if start == None:
                self.init_x = np.random.randint(1,self.size) 
                self.init_y = np.random.randint(1,self.size)
            else:
                self.init_x = start[0] 
                self.init_y = start[1]
            
            if (self.init_x - 1)*size + self.init_y not in dropped:
                cond = False
                
                self.player = (self.init_x,self.init_y)
                
        self._get_state()
                
    def _get_state(self):
        state = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        state[self.goal[0]-1][self.goal[1]-1] = (0, 255, 0)
        for trap in self.traps:
            state[trap[0]-1][trap[1]-1] = (0, 0, 255)
        state[self.player[0]-1][self.player[1]-1] = (255, 175, 0)
        
        img = Image.fromarray(state, 'RGB')
        img = img.resize((128, 128))
        
        self.state = img
        
        grayscale = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(grayscale, (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)
    
    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))
                
    def reset(self):
        self._reset_buffer()
        
        observation = self._get_state()
        self.state_buffer.append(observation)
        
        self.tot_reward = 0
        
        return torch.stack(list(self.state_buffer), 0)
    
    def step(self,action):
        reward = 0
        done = False
        if action == 0:
            if self.player[0] > 1:
                self.player = (self.player[0] - 1,self.player[1])
                if self.player == self.goal:
                    done = True
                    reward = self.g_reward
                elif self.player in self.traps:
                    done = True
                    reward = self.t_penalty
                else:
                    reward = self.m_penalty
            else:
                reward = self.o_penalty
        elif action == 1:
            if self.player[0] < self.size:
                self.player = (self.player[0] + 1,self.player[1])
                if self.player == self.goal:
                    done = True
                    reward = self.g_reward
                elif self.player in self.traps:
                    done = True
                    reward = self.t_penalty
                else:
                    reward = self.m_penalty
            else:
                reward = self.o_penalty
        elif action == 2:
            if self.player[1] > 1:
                self.player = (self.player[0],self.player[1] - 1)
                if self.player == self.goal:
                    done = True
                    reward = self.g_reward
                elif self.player in self.traps:
                    done = True
                    reward = self.t_penalty
                else:
                    reward = self.m_penalty
            else:
                reward = self.o_penalty
        elif action == 3:
            if self.player[1] < self.size:
                self.player = (self.player[0],self.player[1] + 1)
                if self.player == self.goal:
                    done = True
                    reward = self.g_reward
                elif self.player in self.traps:
                    done = True
                    reward = self.t_penalty
                else:
                    reward = self.m_penalty
            else:
                reward = self.o_penalty
        else:
            print("This action is not supported, please select action from 0:up, 1:down, 2:left, 3:right")
        
        observation = self._get_state()
        self.state_buffer.append(observation)
        
        self.tot_reward += reward
        
        if self.tot_reward < -100:
            done = True
        
        return torch.stack(list(self.state_buffer), 0), reward, done
    
    def render(self):
        state = self._get_state()
        cv2.imshow('screen', state.cpu().detach().numpy())
        cv2.waitKey(1) & 0xFF
    
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False
        
    def action_space(self):
        return len(self.actions)
    
    def close(self):
        cv2.destroyAllWindows()
 
###############################################################################

# Testing the environment
# Uncomment to test and render a random walk in a randomly generated maze

###############################################################################


SIZE = 5

NUM_TRAP = 2
TRAP_PENALTY = -10

OUT_PENALTY = -5
MOVE_PENALTY = -1

GOAL_REWARD = 10
       
#Create the environment
maze = Env(size=SIZE, ntraps=NUM_TRAP, args=args)

# Visualize state (colorful image)
img = maze.state
cv2.imwrite("env.png", np.array(img))
cv2.imshow("image", np.array(img))
cv2.waitKey(1) & 0xFF
time.sleep(4)
maze.close()

#Test and render a random walk in the environment
N = 20
for i in range(N):
    action = np.random.randint(0,3)
    state, reward, done = maze.step(action)
    maze.render()
    time.sleep(1)
maze.close()


             
                
                
                
        
    
    
    
    
    
    
    
    
    
    