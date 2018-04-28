import numpy as np
import random
from datetime import datetime
from gridworld import Environment
from agents import RandomAgent
# from tomnet import ToMnet

from torch.optim import Adam
from torch.nn import KLDivLoss

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')


import torch
import torch.nn as nn

import numpy as np

from torch.autograd import Variable



class CharacterNetwork(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_shape):
        super(CharacterNetwork, self).__init__()
        self.input_shape = input_shape #(tuple)
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape

        self.conv = nn.Conv1d(hidden_shape, hidden_shape, input_shape[1])
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=hidden_shape, hidden_size=hidden_shape)
        self.pool = nn.AvgPool1d(hidden_shape)
        self.embed = nn.Linear(in_features=input_shape[0], out_features=output_shape)

    def forward(self, input, hidden_curr):
        # print('Debugging CharNet Forward')
        # print('input.shape:', input.shape)
        out = self.conv(input)
        # print('conv.shape:', out.shape)
        out = self.relu(out).squeeze()
        # print('relu.shape:', out.shape)
        out, hidden_next = self.lstm(out, hidden_curr)
        # print('lstm.shape:', out.shape)
        out = self.pool(out).squeeze()
        # print('pool.shape:', out.shape)
        out = self.embed(out)
        # print('embed.shape:', out.shape)
        return (out, hidden_next)

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_shape)),
                Variable(torch.zeros(1, 1, self.hidden_shape)))

class ToMnet(nn.Module):

    def __init__(self, input_shape, output_shape, feature_planes,
                 character_network=True, character_input_shape=None, character_channels=None,
                 character_embedding_dim=None,
                 mental_network=False, mental_input_shape=None, mental_channels=None,
                 mental_embedding_dim=None,):

        super(ToMnet, self).__init__()
        self.input_shape = input_shape #(tuple)
        self.output_shape = output_shape

        self.character_network = None
        if character_network:
            if (character_input_shape is not None) and (character_channels is not None) and\
                    (character_embedding_dim is not None):
                self.character_network = CharacterNetwork(input_shape=character_input_shape,
                                                          hidden_shape=character_channels,
                                                          output_shape=2)
            else:
                raise Exception('Character Network defined but input and output channel size not defined')

        self.mental_network = None
        if mental_network :
            if character_network:
                if (mental_input_shape is not None) and (mental_channels is not None) and\
                        (mental_embedding_dim is not None):
                    # TODO: Implment mental neural networks
                    pass
                else:
                    raise Exception('Character Network defined but input and output channel size not defined')
            else:
                raise Exception('Mental Network defined without character netwokr!')

        channels = (1 if len(self.input_shape) < 3 else input_shape[2])
        channels += (0 if not character_network else character_embedding_dim)
        channels += (0 if not mental_network else mental_embedding_dim)

        kernal_size = (1 if len(self.input_shape) < 3 else input_shape[1])
        self.conv = nn.Conv1d(channels, feature_planes, kernal_size)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(feature_planes)
        self.linear = nn.Linear(input_shape[0], output_shape)
        # self.logit = lambda v: torch.log(v) - torch.log(torch.ones_like(v) - v)
        self.softmax = nn.LogSoftmax()

        self.epoch = 0
        self.running_loss = []

    def forward(self, input):
        # print('Debugging ToMNet Forward')
        # print('input.shape:', input.shape)
        out = self.conv(input)
        # print('conv.shape:', out.shape)
        out = self.relu(out).view(input.shape[0], 1, -1)
        # print('relu.shape:', out.shape)
        out = self.pool(out).squeeze()
        out = self.linear(out)
        # out = self.logit(out)
        out = self.softmax(out)
        return out

    def train_step(self, example):
        episodes, query_state = example
        trajectories = []

        for state, action in episodes:
            tiled_action = np.zeros((action.shape[0], state.shape[1], state.shape[2]))
            for a in range(action.shape[0]):
                tiled_action[a, :, :] = np.ones((state.shape[1], state.shape[2])) * action[a]
            trajectory = np.vstack((state, tiled_action))
            channels, height, width = trajectory.shape
            trajectories.append(trajectory.reshape(1, height, channels, width))

        trajectories = Variable(torch.Tensor(np.vstack(trajectories)))

        character_hidden = self.character_network.init_hidden()

        trajectory = trajectories[0]
        e_char, character_hidden = self.character_network.forward(trajectory, character_hidden)

        for j in range(1, trajectories.shape[0]):
            trajectory = trajectories[j]
            e_char, character_hidden = self.character_network.forward(trajectory, character_hidden)

        e_char_tiled = e_char.repeat(query_state.shape[1], query_state.shape[2], 1).view(-1, query_state.shape[1], query_state.shape[2])
        tiled_input = torch.cat((Variable(torch.Tensor(query_state)), e_char_tiled)).view(query_state.shape[1], -1, query_state.shape[2])

        return self.forward(tiled_input)



    def train_epoch(self, examples, optimizer, criterion):
        running_loss = 0.0
        for agent in examples:
            optimizer.zero_grad()
            episodes, q_state, target = examples[agent]
            target_v = Variable(torch.Tensor(target))
            out = self.train_step((episodes, q_state))
            loss = criterion(out, target_v)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

        self.running_loss.append(running_loss/len(examples))
        self.epoch += 1


n_agents_per_species = 1000

max_episodes = 10
# this is S(\alpha)
species = [0.01, 0.03,
            0.1,  0.3,
              1,    3]

if __name__ == '__main__':

    # Lets create a sample of 1000 agents in randomly generated POMDPS for each species
    print('Beginning random sampling of Random Agents')
    start_time = datetime.now()
    A = {alpha:{} for alpha in species}
    for alpha in species:
        print('\t Randomly saampling species S(a=%0.2f)'%(alpha))
        for i in range(n_agents_per_species):
            e = Environment(shape=(11, 11), coverage=0.10)
            init_x, init_y = (np.random.randint(0, e.shape[0]-1),
                              np.random.randint(0, e.shape[1]-1))
            while e.grid[init_x, init_y] != 0:
                init_x, init_y = (np.random.randint(0, e.shape[0] - 1),
                                  np.random.randint(0, e.shape[1] - 1))
            agent_i = 'alfie:%0.2f_id:%d'%(alpha, i)
            e.add_agent(RandomAgent(name=agent_i,
                                    pos=(init_x, init_y),
                                    alpha=alpha))

            A[alpha][agent_i] = e
    construct_time = datetime.now() - start_time
    print('Construction completed %d min %d sec'%(int(construct_time.seconds/60), int(construct_time.seconds%60)))
    # We will first try a significantly simplified architecture/state action
    print('Constructing Episodes for each agent')
    start_time = datetime.now()
    training_data = {}
    for alpha in A:
        print('\t Constructing episodes for species S(a=%0.2f)' % (alpha))
        training_data[alpha] = {}
        for agent_i in A[alpha]:
            N_past = np.random.randint(0, max_episodes)
            episodes = []
            for n in range(N_past - 1):
                trajectory = (A[alpha][agent_i].observe_state(),)
                A[alpha][agent_i].update()
                trajectory += (A[alpha][agent_i].agents[agent_i].action_observe(),)
                episodes.append(trajectory)

            query_state = A[alpha][agent_i].observe_state()
            A[alpha][agent_i].update()
            final_action = A[alpha][agent_i].agents[agent_i].action_observe()

            if len(episodes) == 0:
                episodes.append((np.zeros_like(A[alpha][agent_i].observe_state()),
                                 np.zeros_like(A[alpha][agent_i].agents[agent_i].action_observe())))
            training_data[alpha][agent_i] = (episodes, query_state, final_action)

    construct_time = datetime.now() - start_time
    print('Construction completed %d min %d sec' % (int(construct_time.seconds / 60), int(construct_time.seconds % 60)))


    episodes = []
    while len(episodes) == 0:
        r_alpha = random.choice(list(A.keys()))
        r_agent = random.choice(list(training_data[r_alpha].keys()))
        print('r_agent:', r_agent)
        episodes, query_state, final_action = training_data[r_alpha][r_agent]
        print('len episodes:', len(episodes))

    tomnets = {a:ToMnet(input_shape=(11, 11, 3),
                        output_shape=5,
                        feature_planes=32,
                        character_input_shape=(11, 11),
                        character_channels=8,
                        character_embedding_dim=2) for a in A}
    optimizers = {a:Adam(tomnets[a].parameters(), lr=1e-4) for a in A}
    criterion = KLDivLoss()

    num_epochs = 10
    for a in tomnets:

        for epoch in range(num_epochs):
            print('Alpha:%0.3f\tEpoch:%d'%(a, epoch))
            start_time = datetime.now()
            tomnets[a].train_epoch(training_data[a], optimizers[a], criterion)
            epoch_time = datetime.now() - start_time
            print('Epoch Loss: %0.4f'%(tomnets[a].running_loss[-1]))
            print('Epoch completed %d min %d sec' % (
            int(epoch_time.seconds / 60), int(epoch_time.seconds % 60)))
            print('-'*20)
        plt.plot(list(range(num_epochs)), tomnets[a].running_loss)
        plt.title('S(%0.3f)'%(a))
        plt.xlabel('EPochs')
        plt.ylabel('KL Divergence')
        plt.show()
        print('\n\n')

    print('Completed Learning...')

    print('Beginning random sampling of Random Agents for ')
    start_time = datetime.now()
    A_test = {alpha:{} for alpha in species}
    for alpha in species:
        print('\t Randomly saampling species S(a=%0.2f)'%(alpha))
        for i in range(int(n_agents_per_species * .1)):
            e = Environment(shape=(11, 11), coverage=0.10)
            init_x, init_y = (np.random.randint(0, e.shape[0]-1),
                              np.random.randint(0, e.shape[1]-1))
            while e.grid[init_x, init_y] != 0:
                init_x, init_y = (np.random.randint(0, e.shape[0] - 1),
                                  np.random.randint(0, e.shape[1] - 1))
            agent_i = 'alfie:%0.2f_id:%d'%(alpha, i)
            e.add_agent(RandomAgent(name=agent_i,
                                    pos=(init_x, init_y),
                                    alpha=alpha))

            A_test[alpha][agent_i] = e
    construct_time = datetime.now() - start_time
    print('Construction completed %d min %d sec'%(int(construct_time.seconds/60), int(construct_time.seconds%60)))
    # We will first try a significantly simplified architecture/state action
    print('Constructing Episodes for each test agent')
    start_time = datetime.now()
    testing_data = {}
    for alpha in A_test:
        print('\t Constructing episodes for species S(a=%0.2f)' % (alpha))
        testing_data[alpha] = {}
        for agent_i in A_test[alpha]:
            N_past = 10
            episodes = []
            for n in range(N_past - 1):
                trajectory = (A_test[alpha][agent_i].observe_state(),)
                A_test[alpha][agent_i].update()
                trajectory += (A_test[alpha][agent_i].agents[agent_i].action_observe(),)
                episodes.append(trajectory)

            query_state = A_test[alpha][agent_i].observe_state()
            A_test[alpha][agent_i].update()
            final_action = A_test[alpha][agent_i].agents[agent_i].action_observe()

            if len(episodes) == 0:
                episodes.append((np.zeros_like(A_test[alpha][agent_i].observe_state()),
                                 np.zeros_like(A_test[alpha][agent_i].agents[agent_i].action_observe())))
            testing_data[alpha][agent_i] = (episodes, query_state, final_action)

    construct_time = datetime.now() - start_time
    print('Construction completed %d min %d sec' % (int(construct_time.seconds / 60), int(construct_time.seconds % 60)))


    # tomnets = {alpha:ToMnet(input_shape=) for alpha in A}


