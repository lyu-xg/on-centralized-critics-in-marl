import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2


class EnvGoTogether(object):
    def __init__(self, size=9):
        self.map_size = size
        self.state_space = spaces.Box(
            low=-255, high=255, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-255, high=255, shape=(3*3*3,), dtype=np.float32
        )
        self.occupancy = np.zeros((self.map_size, self.map_size))
        for i in range(self.map_size):
            self.occupancy[0][i] = 1
            self.occupancy[self.map_size - 1][i] = 1
            self.occupancy[i][0] = 1
            self.occupancy[i][self.map_size - 1] = 1
        self.agt1_pos = [self.map_size - 3, 1]
        self.agt2_pos = [self.map_size - 2, 2]
        self.goal_pos = [1, self.map_size - 2]
        self.n_agents = 2

    def reset(self):
        self.i_step = 0
        self.occupancy = np.zeros((self.map_size, self.map_size))
        for i in range(self.map_size):
            self.occupancy[0][i] = 1
            self.occupancy[self.map_size - 1][i] = 1
            self.occupancy[i][0] = 1
            self.occupancy[i][self.map_size - 1] = 1
        self.agt1_pos = [self.map_size - 3, 1]
        self.agt2_pos = [self.map_size - 2, 2]
        self.goal_pos = [1, self.map_size - 2]
        return self.get_obs()

    def get_state(self):
        state = np.zeros((1, 4))
        state[0, 0] = self.agt1_pos[0] / self.map_size
        state[0, 1] = self.agt1_pos[1] / self.map_size
        state[0, 2] = self.agt2_pos[0] / self.map_size
        state[0, 3] = self.agt2_pos[1] / self.map_size
        return state.reshape(-1)

    def step(self, action_list):
        self.i_step += 1
        reward = 0.0
        # agent1 move
        if action_list[0] == 0:  # move up
            if (
                self.occupancy[self.agt1_pos[0] - 1][self.agt1_pos[1]] != 1
            ):  # if can move
                self.agt1_pos[0] = self.agt1_pos[0] - 1
        elif action_list[0] == 1:  # move down
            if (
                self.occupancy[self.agt1_pos[0] + 1][self.agt1_pos[1]] != 1
            ):  # if can move
                self.agt1_pos[0] = self.agt1_pos[0] + 1
        elif action_list[0] == 2:  # move left
            if (
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] - 1] != 1
            ):  # if can move
                self.agt1_pos[1] = self.agt1_pos[1] - 1
        elif action_list[0] == 3:  # move right
            if (
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] + 1] != 1
            ):  # if can move
                self.agt1_pos[1] = self.agt1_pos[1] + 1

        # agent2 move
        if action_list[1] == 0:  # move up
            if (
                self.occupancy[self.agt2_pos[0] - 1][self.agt2_pos[1]] != 1
            ):  # if can move
                self.agt2_pos[0] = self.agt2_pos[0] - 1
        elif action_list[1] == 1:  # move down
            if (
                self.occupancy[self.agt2_pos[0] + 1][self.agt2_pos[1]] != 1
            ):  # if can move
                self.agt2_pos[0] = self.agt2_pos[0] + 1
        elif action_list[1] == 2:  # move left
            if (
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1] - 1] != 1
            ):  # if can move
                self.agt2_pos[1] = self.agt2_pos[1] - 1
        elif action_list[1] == 3:  # move right
            if (
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1] + 1] != 1
            ):  # if can move
                self.agt2_pos[1] = self.agt2_pos[1] + 1

        if self.agt1_pos == self.goal_pos and self.agt2_pos == self.goal_pos:
            reward = reward + 10

        if (
            self.sqr_dist(self.agt1_pos, self.agt2_pos) <= 1
            or self.sqr_dist(self.agt1_pos, self.agt2_pos) > 9
        ):
            reward = reward - 0.5
        return self.get_obs(), reward, reward > 0 or self.i_step >= 2000

    def sqr_dist(self, pos1, pos2):
        return (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (
            pos1[1] - pos2[1]
        )

    def get_global_obs(self):
        obs = np.zeros((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.occupancy[i][j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
        obs[self.agt1_pos[0], self.agt1_pos[1], 0] = 1.0
        obs[self.agt1_pos[0], self.agt1_pos[1], 1] = 0.0
        obs[self.agt1_pos[0], self.agt1_pos[1], 2] = 0.0
        obs[self.agt2_pos[0], self.agt2_pos[1], 0] = 0.0
        obs[self.agt2_pos[0], self.agt2_pos[1], 1] = 0.0
        obs[self.agt2_pos[0], self.agt2_pos[1], 2] = 1.0
        obs[self.goal_pos[0], self.goal_pos[1], 0] = 0.0
        obs[self.goal_pos[0], self.goal_pos[1], 1] = 1.0
        obs[self.goal_pos[0], self.goal_pos[1], 2] = 0.0
        return obs
    
    def get_obs(self):
        return [
            self.get_local_obs(self.agt1_pos, self.agt2_pos),
            self.get_local_obs(self.agt2_pos, self.agt1_pos)
        ]
    
    def get_local_obs(self, agt_pos, teammate_pos, flat=True):
        obs = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                if (
                    self.occupancy[agt_pos[0] - 1 + i][agt_pos[1] - 1 + j]
                    == 0
                ):
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                d_x = self.goal_pos[0] - agt_pos[0]
                d_y = self.goal_pos[1] - agt_pos[1]
                if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                    obs[1 + d_x, 1 + d_y, 0] = 0.0
                    obs[1 + d_x, 1 + d_y, 1] = 1.0
                    obs[1 + d_x, 1 + d_y, 2] = 0.0
                d_x = teammate_pos[0] - agt_pos[0]
                d_y = teammate_pos[1] - agt_pos[1]
                if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                    obs[1 + d_x, 1 + d_y, 0] = 0.0
                    obs[1 + d_x, 1 + d_y, 1] = 0.0
                    obs[1 + d_x, 1 + d_y, 2] = 1.0
        obs[1, 1, 0] = 1.0
        obs[1, 1, 1] = 0.0
        obs[1, 1, 2] = 0.0
        if flat: obs = obs.reshape(-1)
        return obs

    def plot_scene(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.get_global_obs())
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def render(self):
        obs = self.get_global_obs()
        enlarge = 30
        new_obs = np.ones((self.map_size * enlarge, self.map_size * enlarge, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):

                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(
                        new_obs,
                        (j * enlarge, i * enlarge),
                        (j * enlarge + enlarge, i * enlarge + enlarge),
                        (0, 0, 0),
                        -1,
                    )
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(
                        new_obs,
                        (j * enlarge, i * enlarge),
                        (j * enlarge + enlarge, i * enlarge + enlarge),
                        (0, 0, 255),
                        -1,
                    )
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(
                        new_obs,
                        (j * enlarge, i * enlarge),
                        (j * enlarge + enlarge, i * enlarge + enlarge),
                        (0, 255, 0),
                        -1,
                    )
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0:
                    cv2.rectangle(
                        new_obs,
                        (j * enlarge, i * enlarge),
                        (j * enlarge + enlarge, i * enlarge + enlarge),
                        (255, 0, 0),
                        -1,
                    )
        cv2.imshow("image", new_obs)
        cv2.waitKey(100)
