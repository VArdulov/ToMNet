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




