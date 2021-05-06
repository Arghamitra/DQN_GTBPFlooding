import random
import time
from collections import deque, defaultdict, namedtuple
import numpy as np
import pandas as pd
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from environment import Environment
import param
import pickle
import sys

env_name = 'BP'
single_nn = False

#elif env_name == 'BP':
#env = Environment(trial = 0,  status = 'train', state_asnp= True)
if env_name == 'gym':
    env = gym.make('LunarLander-v2')

# Agent that takes random actions
max_episodes = 100

scores = []
if env_name == 'gym':
    actions = range(env.action_space.n)
#elif env_name == 'BP':
    #actions = range(param.m)


# Use cuda if available else use cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add experience"""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert to torch tensors
        states = torch.from_numpy(
            np.vstack([experience.state for experience in experiences if experience is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([experience.action for experience in experiences if experience is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([experience.reward for experience in experiences if experience is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([experience.next_state for experience in experiences if experience is not None])).float().to(
            device)
        # Convert done from boolean to int
        dones = torch.from_numpy(
            np.vstack([experience.done for experience in experiences if experience is not None]).astype(
                np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


BUFFER_SIZE = int(1e5)  # Replay memory size
BATCH_SIZE = 64  # Number of experiences to sample from memory
GAMMA = 0.99  # Discount factor
TAU = 1e-3  # Soft update parameter for updating fixed q network
LR = 1e-4  # Q Network learning rate
UPDATE_EVERY = 1  # How often to update Q network


class DQNAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # Initialize Q and Fixed Q networks
        self.q_network = QNetwork(state_size, action_size, seed).to(device)
        self.fixed_network = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters())
        # Initiliase memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.timestep = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.timestep += 1
        if self.timestep % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                sampled_experiences = self.memory.sample()
                self.learn(sampled_experiences)

    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences
        # Get the action with max Q value
        action_values = self.fixed_network(next_states).detach()
        # Notes
        # tensor.max(1)[0] returns the values, tensor.max(1)[1] will return indices
        # unsqueeze operation --> np.reshape
        # Here, we make it from torch.Size([64]) -> torch.Size([64, 1])
        max_action_values = action_values.max(1)[0].unsqueeze(1)

        # If done just use reward, else update Q_target with discounted action values
        Q_target = rewards + (GAMMA * max_action_values * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)

        # Calculate loss
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        # backward pass
        loss.backward()
        # update weights
        self.optimizer.step()

        # Update fixed weights
        self.update_fixed_network(self.q_network, self.fixed_network)

    def update_fixed_network(self, q_network, fixed_network):

        for source_parameters, target_parameters in zip(q_network.parameters(), fixed_network.parameters()):
            target_parameters.data.copy_(TAU * source_parameters.data + (1.0 - TAU) * target_parameters.data)

    def act(self, state, eps=0.0):

        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # set the network into evaluation mode
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            # Back to training mode
            self.q_network.train()
            action = np.argmax(action_values.cpu().data.numpy())
            return action

    def checkpoint(self, filename):
        torch.save(self.q_network.state_dict(), filename)



BUFFER_SIZE = int(1e5) # Replay memory size
BATCH_SIZE = 64         # Number of experiences to sample from memory
GAMMA = 0.99            # Discount factor
TAU = 1e-3              # Soft update parameter for updating fixed q network
LR = 1e-4               # Q Network learning rate
UPDATE_EVERY = 1        # How often to update Q network

MAX_EPISODES = 1500  # Max number of episodes to play
MAX_STEPS = 101     # Max steps allowed in a single episode/play
ENV_SOLVED = 200     # MAX score at which we consider environment to be solved
PRINT_EVERY = 100    # How often to print the progress

# Epsilon schedule

EPS_START = 1.0      # Default/starting value of eps
EPS_DECAY = 0.999    # Epsilon decay rate
EPS_MIN = 0.01       # Minimum epsilon

EPS_DECAY_RATES = [0.9, 0.99, 0.999, 0.9999]


for decay_rate in EPS_DECAY_RATES:
    test_eps = EPS_START
    eps_list = []
    for _ in range(MAX_EPISODES):
        test_eps = max(test_eps * decay_rate, EPS_MIN)
        eps_list.append(test_eps)



if env_name == 'gym':
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print('State size: {}, action size: {}'.format(state_size, action_size))
#elif env_name == 'BP':
    #state_size = param.m
    #action_size = param.m




if env_name == 'gym':
    dqn_agent = DQNAgent(state_size, action_size, seed=0)

def training():
    print('----------TRAIN---------')
    print('Episode num: ', MAX_EPISODES)
    start = time.time()
    scores = []
    episodes = []
    results = []
    scores_window = deque(maxlen=100)
    eps = EPS_START

    if env_name == 'BP':
        actions = range(param.m)
        state_size = param.m
        action_size = param.m

        dqn_agentS = []
        if single_nn:
            dqn_agent = DQNAgent(state_size, action_size, seed=0)
        else:
            for t in range(MAX_STEPS):
                dqn_agent = DQNAgent(state_size, action_size, seed=0)
                dqn_agentS.append(dqn_agent)

    for episode in range(1, MAX_EPISODES + 1):

        if env_name == 'BP':
            env = Environment(episode, status='train', state_asnp=True)

        state = env.reset()
        score = 0
        for t in range(MAX_STEPS):

            if single_nn:
                action = dqn_agent.act(state, eps)
            else:
                action = dqn_agentS[t].act(state, eps)
            next_state, reward, done, info = env.step(action)

            if single_nn:
                dqn_agent.step(state, action, reward, next_state, done)
            else:
                dqn_agentS[t].step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

            eps = max(eps * EPS_DECAY, EPS_MIN)
            if episode % PRINT_EVERY == 0:
                mean_score = np.mean(scores_window)
                print('\r Progress {}/{}, average score:{:.2f}'.format(episode, MAX_EPISODES, mean_score), end="")

        result = env.CodeRunner(episode)
        results.append(result)
        scores_window.append(score)
        scores.append(score)
        episodes.append(episode)
        print('epi ', episode, ' score ', score, 'avg score ', np.mean(scores))

    plt.figure(figsize=(10,6))
    plt.plot(episodes, scores)
    plt.plot(pd.Series(scores).rolling(100).mean())
    plt.title('DQN Training')
    plt.xlabel('# of episodes')
    plt.ylabel('score')
    #plt.show()
    plt.grid()
    trimester = time.strftime("_%Y_%m_%d-%H__%M_%S")
    plt.savefig('./pic/TR_DQN_' + str(MAX_EPISODES) + '_' + (trimester) + '.png')
    env.show_result(results, MAX_EPISODES)
    print('TRAINING {} seconds'.format(time.time() - start))


    print('----------TEST---------')
    print('Episode num: ', MAX_EPISODES)

    start = time.time()
    scores = []
    episodes = []
    results = []
    scores_window = deque(maxlen=100)
    eps = 0
    episode = 0

    for episodeT in range(1, MAX_EPISODES + 1):

        if env_name == 'BP':
            env = Environment(episodeT, status='train', state_asnp=True)
            actions = range(param.m)
            state_size = param.m
            action_size = param.m

        state = env.reset()
        score = 0
        for tT in range(MAX_STEPS):

            if single_nn:
                action = dqn_agent.act(state, eps)
            else:
                action = dqn_agentS[tT].act(state, eps)
            next_state, reward, done, info = env.step(action)
            """
            if single_nn:
                dqn_agent.step(state, action, reward, next_state, done)
            else:
                dqn_agentS[t].step(state, action, reward, next_state, done)
            """
            state = next_state
            score += reward
            if done:
                break

            if episodeT % PRINT_EVERY == 0:
                mean_score = np.mean(scores_window)
                print('\r Progress {}/{}, average score:{:.2f}'.format(episode, MAX_EPISODES, mean_score), end="")

        result = env.CodeRunner(episodeT)
        results.append(result)
        scores_window.append(score)
        scores.append(score)
        episodes.append(episodeT)
        print('epi ', episodeT, ' score ', score, 'avg score ', np.mean(scores))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, scores)
    plt.plot(pd.Series(scores).rolling(100).mean())
    plt.title('DQN Testing')
    plt.xlabel('# of episodes')
    plt.ylabel('score')
    #plt.show()
    plt.grid()
    trimester = time.strftime("_%Y_%m_%d-%H__%M_%S")
    plt.savefig('./pic/TS_DQN_' + str(MAX_EPISODES) + '_' + (trimester) + '.png')
    env.show_result(results, MAX_EPISODES)
    print('TESTING {} seconds'.format(time.time() - start))

if __name__ == '__main__':
    training()
    #testing()