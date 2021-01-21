# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 17:56:35 2020

@author: abdel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from moviepy.editor import ImageSequenceClip
import pandas as pd

import os

from my_env import Maze

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, state):
        # ====================================================
        # YOUR IMPLEMENTATION HERE 
        #
        
        x = self.fc1(state)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        Q = self.fc3(x)
        
        # ====================================================
        return Q

    def select_greedyaction(self, state):
        with torch.no_grad():
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            #
            
            Q = self.forward(state)
            
            #Greedy action
            action_index = Q.max(1)[1].unsqueeze(1)
            
            # ====================================================
        return action_index.item()

##############################################################################

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, sample):
        """Saves a transition.
            sample is a tuple (state, next_state, action, reward, done)
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(len(self.memory), batch_size)
        samples = random.sample(self.memory, batch_size)
        return map(np.asarray, zip(*samples))

    def __len__(self):
        return len(self.memory)

##############################################################################

def eval_dqn(env, qnet, n_sim=10):
    """
    Monte Carlo evaluation of DQN agent
    """
    rewards = np.zeros(n_sim)
    copy_env = deepcopy(env) # Important!
    # Loop over number of simulations
    for sim in range(n_sim):
        state = copy_env.reset()
        done = False
        while not done:
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = qnet.select_greedyaction(tensor_state)
            done, reward = copy_env.step(action)
            next_state = copy_env.player
            # update sum of rewards
            rewards[sim] += reward
            state = next_state
    return rewards

##############################################################################

# Discount factor
GAMMA = 0.99
EVAL_EVERY = 2

# Batch size
BATCH_SIZE = 256
# Capacity of the replay buffer
BUFFER_CAPACITY = 30000
# Update target net every 'C' episodes
UPDATE_TARGET_EVERY = 10

# Initial value of epsilon
EPSILON_START = 1
# Parameter to decrease epsilon
DECREASE_EPSILON = 200
# Minimum value of epislon
EPSILON_MIN = 0.05

# Number of training episodes
N_EPISODES = 250

##############################################################################
# Learning rate (was reduced from 0.1 to 0.001 which gave better results)
LEARNING_RATE = 0.001
##############################################################################

SIZE = 10

NUM_TRAP = 3
TRAP_PENALTY = -10

OUT_PENALTY = -100 
MOVE_PENALTY = -1

GOAL_REWARD = 10

env = Maze(size=SIZE, 
            ntraps=NUM_TRAP, m_penalty=MOVE_PENALTY,o_penalty=OUT_PENALTY,
            t_penalty=TRAP_PENALTY,g_reward=GOAL_REWARD)

# initialize replay buffer
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

# create network and target network
obs_size = 2
n_actions = 4
print("n_actions: ", n_actions)


number_iter = 1
for iterat in range(number_iter):
    # ====================================================
    # YOUR IMPLEMENTATION HERE 
    # Define networks
    #
    #The main network
    q_net = QNet(obs_size = obs_size,n_actions = n_actions).to(device)
    
    #The target network initialized with the same weights
    target_net = QNet(obs_size = obs_size,n_actions = n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    # ====================================================
    
    # objective and optimizer
    optimizer = optim.Adam(params=q_net.parameters(), lr=LEARNING_RATE)
    #optimizer = optim.RMSprop(q_net.parameters())
    
    # Algorithm
    state = env.reset()
    epsilon = EPSILON_START
    ep = 0
    total_time = 0
    learn_steps = 0
    episode_reward = 0
    
    episode_rewards = np.zeros((N_EPISODES, 3))
    while ep < N_EPISODES:
        # ====================================================
        # YOUR IMPLEMENTATION HERE 
        # sample epsilon-greedy action
        #
        p = random.random()
        if p < epsilon:
            #Select an action with uniform probability
            action = np.random.randint(low = 0, high = n_actions)
        else:
            #Select greedy_action
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = q_net.select_greedyaction(tensor_state)
        # ====================================================
    
        done, reward = env.step(action)
        next_state = env.player
        episode_reward += reward
        total_time += 1
        
        # ====================================================
        # YOUR IMPLEMENTATION HERE 
        # add sample to buffer
        #
        sample_tuple = (state,next_state,action,reward,done)
        replay_buffer.push(sample_tuple)
        # ====================================================
    
    
        if len(replay_buffer) > BATCH_SIZE:
            learn_steps += 1
            # UPDATE MODEL
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # get batch
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = replay_buffer.sample(BATCH_SIZE)
            # ====================================================
    
    
            batch_state = torch.FloatTensor(batch_state).to(device)
            batch_next_state = torch.FloatTensor(batch_next_state).to(device)
            batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
            batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)
    
            with torch.no_grad():
                # ====================================================
                # YOUR IMPLEMENTATION HERE 
                # build target (recall that we conseder the Q function
                # in the next state only if not terminal, ie. done != 1)
                # (1- done) * value_next
                #
                # targets = ...
                
                #DQN
                
                #Greedy action from the target network
                target_Q = target_net(batch_next_state)
                value_next = target_Q.max(dim = 1, keepdim = False)[0]
                
                #Eliminate terminal states
                mask = (1 - batch_done).reshape(value_next.shape)
                value_next = (value_next * mask).unsqueeze(1).to(device)
                
                #Target values
                targets = batch_reward + GAMMA * value_next
                # ====================================================
    
            #current predictions
            q_net.train()
            values = q_net(batch_state).gather(1, batch_action.long())
    
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # compute loss and update model (loss and optimizer)
            optimizer.zero_grad()
            loss = F.mse_loss(values,targets)
            loss.backward()
            optimizer.step()
            # ====================================================
    
            if epsilon > EPSILON_MIN:
                epsilon -= (EPSILON_START - EPSILON_MIN) / DECREASE_EPSILON
        
    
        # ====================================================
        # YOUR IMPLEMENTATION HERE 
        # update target network
        if learn_steps % UPDATE_TARGET_EVERY == 0:
            target_net.train()
            target_net.load_state_dict(q_net.state_dict())
            target_net.eval()
        # ====================================================
    
        state = next_state
        if done:
            mean_rewards = -1
            if (ep+1) % EVAL_EVERY == 0:
                # evaluate current policy
                q_net.eval()
                rewards = eval_dqn(env, q_net)
                mean_rewards = np.mean(rewards)
                print("episode =", ep, ", reward = ", np.round(np.mean(rewards),2), ", obs_rew = ", episode_reward)
            
            episode_rewards[ep] = [total_time, episode_reward, mean_rewards]
            state = env.reset()
            ep += 1
            episode_reward = 0
    
    #Save current run's results 
    
    # Create folder
    folder = 'dqn'
    if not os.path.isdir(folder):
        os.makedirs(folder)
    
    #set path
    path = folder+'/run'+str(iterat)+'.csv'
    
    #save to csv
    df = pd.DataFrame(episode_rewards)
    df.to_csv(path)

###################################################################
# VISUALIZATION
###################################################################
    
frames = []
for episode in range(3):
    done = False
    state = env.reset()
    frame = env.render()
    frames.append(np.asarray( frame, dtype="int32" ))
    while not done:
        tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = q_net.select_greedyaction(tensor_state)
        print(action)
        done, reward = env.step(action)
        state = env.player
        frame = env.render()
        frames.append(np.asarray( frame, dtype="int32" ))  

clip = ImageSequenceClip(frames, fps=20)
clip.write_gif('test.gif', fps=20)

plt.figure(1)
plt.title('Performance over learning (DQN)')
plt.plot(episode_rewards[:,0], episode_rewards[:,1])
plt.xlabel('time steps')
plt.ylabel('total reward')

plt.figure(2)
plt.title('Performance on Test Env (DQN)')
xv = np.arange(EVAL_EVERY-1, N_EPISODES+1, EVAL_EVERY)
plt.plot(episode_rewards[xv, 0], episode_rewards[xv, 2], ':o')
plt.xlabel('time steps')
plt.ylabel('expected total reward (greedy policy)')
plt.show()






