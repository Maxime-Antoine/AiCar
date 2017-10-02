import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


class Network(nn.Module):
    NB_HIDDEN_LAYER_NEURONS = 30

    def __init__(self, input_size, nb_actions):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_actions = nb_actions
        self.fc1 = nn.Linear(input_size, self.NB_HIDDEN_LAYER_NEURONS)
        self.fc2 = nn.Linear(self.NB_HIDDEN_LAYER_NEURONS, nb_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state)) # rectifier function
        q_values = self.fc2(x)
        return q_values


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class Dqn(object):

    MEMORY_SIZE = 100000
    SAVED_FILE_NAME = 'last_brain.pth'
    TEMPERATURE = 10
    LEARNING_RATE = 0.01
    SAMPLE_SIZE_TO_LEARN_FROM = 100
    MAX_REWARD_WINDOW_SIZE = 1000

    def __init__(self, input_size, nb_actions, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_actions)
        self.memory = ReplayMemory(self.MEMORY_SIZE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        var_state = Variable(state, volatile=True)
        probs = F.softmax(self.model(var_state) * self.TEMPERATURE)
        action = probs.multinomial()
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > self.SAMPLE_SIZE_TO_LEARN_FROM:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(self.SAMPLE_SIZE_TO_LEARN_FROM)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > self.MAX_REWARD_WINDOW_SIZE:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                   }, self.SAVED_FILE_NAME)

    def load(self):
        if os.path.isfile(self.SAVED_FILE_NAME):
            print("=> Loading checkpoint...")
            checkpoint = torch.load(self.SAVED_FILE_NAME)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded !")
        else:
            print("No checkpoint to load")
