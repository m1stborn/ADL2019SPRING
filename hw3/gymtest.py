import gym
import argparse
from environment import Environment

import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

use_cuda = torch.cuda.is_available()

def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--train_mario', action='store_true', help='whether train mario')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--test_mario', action='store_true', help='whether test mario')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')

    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args
args = parse()
env_name = 'AssaultNoFrameskip-v0'
env = Environment(env_name, args, atari_wrapper=True)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


memory = ReplayMemory(10000)

from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# # 跑 200 個 episode，每個 episode 都是一次任務嘗試
# for i_episode in range(1):
#     observation = env.reset() # 讓 environment 重回初始狀態
#     print(observation.shape)
#     rewards = 0 # 累計各 episode 的 reward
#     for t in range(1): # 設個時限，每個 episode 最多跑 250 個 action
#         # env.render() # 呈現 environment
#
#         # Key section
#         action = env.action_space.sample() # 在 environment 提供的 action 中隨機挑選
#         observation, reward, done, info = env.step(action) # 進行 action，environment 返回該 action 的 reward 及前進下個 state
#         memory.push(observation, 0, next_state, reward)
#         rewards += reward # 累計 reward
#
#         if done: # 任務結束返回 done = True
#             print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
#             break
#     print('Episode not done after {} timesteps, total rewards {}'.format(t+1, rewards))

#
#
def update(self):
    # TODO:
    # To update model, we sample some stored experiences as training examples.

    # TODO:
    # Compute Q(s_t, a) with your model.

    with torch.no_grad():
        # TODO:
        # Compute Q(s_{t+1}, a) for all next states.
        # Since we do not want to backprop through the expected action values,
        # use torch.no_grad() to stop the gradient from Q(s_{t+1}, a)
        print(0)

    # TODO:
    # Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
    # You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it is the terminal state.

    # TODO:
    # Compute temporal difference loss

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()
def train(env):
    episodes_done_num = 0 # passed episodes
    total_reward = 0 # compute average reward
    loss = 0
    for i in range(1):
        state = env.reset()
        # State: (84,84,4) --> (1,4,84,84)
        state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
        state = state.cuda() if use_cuda else state
        # print(state.size())

        done = False
        for i in range(2):
            # select and perform action
            # action = make_action(state)
            action = env.env.action_space.sample()
            # next_state, reward, done, _ = env.step(action[0, 0].data.item())
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # process new state
            next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
            next_state = next_state.cuda() if use_cuda else next_state
            if done:
                next_state = None

            # TODO:
            # store the transition in memory
            memory.push(state, action, next_state, reward)
            # move to the next state
            state = next_state

    # print(memory.sample(2)[1][0].sum())
# print(env.get_action_space())

train(env)

transitions = memory.sample(2)
batch = Transition(*zip(*transitions))

print(batch)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
#
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
