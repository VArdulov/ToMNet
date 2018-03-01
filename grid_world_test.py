import numpy as np
from gridworld import Environment
from agents import RandomAgent

agent_names = [('alfie', 1e-2),
               ('betty', 1/3),
               ('chuck', 1),
               ('danny', 3)]

grid = np.matrix([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                  [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                  [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                  [-1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                  [-1,  0,  0, -1,  0,  0,  0,  0,  0,  0, -1],
                  [-1,  0,  0, -1,  0,  0,  0,  0,  0,  0, -1],
                  [-1,  0,  0, -1,  0,  0,  0,  0,  0,  0, -1],
                  [-1,  0,  0, -1,  0,  0, -1, -1, -1, -1, -1],
                  [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                  [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

if __name__ == '__main__':

    e = Environment(grid=grid)
    for i, j in enumerate(agent_names):
        name, sparsitiy = j
        placed = False
        while not placed:
            x = np.random.randint(0, grid.shape[0] - 1)
            y = np.random.randint(0, grid.shape[1] - 1)
            if grid[x, y] == 0:
                r_color = np.array([0, (i+1)/len(agent_names), 0])
                print(name, '\t->', r_color)
                e.add_agent(RandomAgent(name,
                                        pos=(x,y),
                                        color=r_color,
                                        alpha=sparsitiy))
                placed = True
    e.show_image()
    e.update(10)
    e.show_image()