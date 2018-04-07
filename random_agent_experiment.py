import numpy as np
import random
from datetime import datetime
from gridworld import Environment
from agents import RandomAgent
from tomnet import ToMnet

from torch.optim import Adam
from torch.nn import KLDivLoss

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')



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


