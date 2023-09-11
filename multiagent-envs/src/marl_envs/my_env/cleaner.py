import numpy as np
from gym import spaces
from .maze import Maze
import random
import cv2


class Cleaner(object):
    def __init__(self, n_agent=2, map_size=13, seed=5):
        self.map_size = map_size
        self.seed = seed
        self.occupancy = self.generate_maze(seed)
        self.n_agent = n_agent
        self.agt_pos_list = []
        self.observation_space = spaces.Box(low=-255, high=255, shape=(169,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.state_space = spaces.Discrete(1) # dummy state_space
        for i in range(self.n_agent):
            self.agt_pos_list.append([1, 1])

    def generate_maze(self, seed):
        symbols = {
            # default symbols
            'start': 'S',
            'end': 'X',
            'wall_v': '|',
            'wall_h': '-',
            'wall_c': '+',
            'head': '#',
            'tail': 'o',
            'empty': ' '
        }
        maze_obj = Maze(int((self.map_size - 1) / 2),
                        int((self.map_size - 1) / 2), seed, symbols, 1)
        grid_map = maze_obj.to_np()
        for i in range(self.map_size):
            for j in range(self.map_size):
                if grid_map[i][j] == 0:
                    grid_map[i][j] = 2
        return grid_map

    def step(self, action_list):
        reward = 0.0
        self.i_step += 1
        for i in range(len(action_list)):
            if action_list[i] == 0:     # up
                # if can move
                if self.occupancy[self.agt_pos_list[i][0] - 1][self.agt_pos_list[i][1]] != 1:
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] - 1
            if action_list[i] == 1:     # down
                # if can move
                if self.occupancy[self.agt_pos_list[i][0] + 1][self.agt_pos_list[i][1]] != 1:
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] + 1
            if action_list[i] == 2:     # left
                # if can move
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] - 1] != 1:
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] - 1
            if action_list[i] == 3:     # right
                # if can move
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] + 1] != 1:
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] + 1
            # if the spot is dirty
            if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]] == 2:
                self.occupancy[self.agt_pos_list[i]
                               [0]][self.agt_pos_list[i][1]] = 0
                reward = reward + 1
        terminal = self.i_step >= 200
        return action_list, self.get_observation(), [float(reward)]*self.n_agent, [terminal]*self.n_agent, [1]*self.n_agent, {}

    def get_global_obs(self):
        obs = np.zeros((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.occupancy[i, j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                if self.occupancy[i, j] == 2:
                    obs[i, j, 0] = 0.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 0.0
        for i in range(self.n_agent):
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 0] = 1.0
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 1] = 0.0
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 2] = 0.0
        return obs

    def reset(self):
        self.i_step = 0
        self.occupancy = self.generate_maze(self.seed)
        self.agt_pos_list = []
        for i in range(self.n_agent):
            self.agt_pos_list.append([1, 1])
        return self.get_observation()

    def get_observation(self):
        obs = self.occupancy.copy()
        obs[self.agt_pos_list[0][0], self.agt_pos_list[0][1]] = 3
        obs[self.agt_pos_list[1][0], self.agt_pos_list[1][1]] = 4
        obs = obs / 4
        return [obs.reshape(-1) for _ in range(self.n_agent)]

    def render(self):
        obs = self.get_global_obs()
        enlarge = 5
        new_obs = np.ones((self.map_size*enlarge, self.map_size*enlarge, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i *
                                                                        enlarge + enlarge, j * enlarge + enlarge), (0, 0, 0), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i *
                                                                        enlarge + enlarge, j * enlarge + enlarge), (0, 0, 255), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i *
                                                                        enlarge + enlarge, j * enlarge + enlarge), (0, 255, 0), -1)
        cv2.imshow('image', new_obs)
        cv2.waitKey(10)
