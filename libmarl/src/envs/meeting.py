import numpy as np
import gym
from gym import spaces
from numpy.random import randint
from configs import get_n_agents, get_ct_dim

NORTH = np.array([0, 1])
SOUTH = np.array([0, -1])
WEST = np.array([-1, 0])
EAST = np.array([1, 0])
STAY = np.array([0, 0])

TRANSLATION_TABLE = [
    # [left, intended_direction, right]
    [WEST, NORTH, EAST],
    [EAST, SOUTH, WEST],
    [SOUTH, WEST, NORTH],
    [NORTH, EAST, SOUTH],
    [STAY, STAY, STAY],
]

DIRECTION = np.array([[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

ACTIONS = ["NORTH", "SOUTH", "WEST", "EAST", "STAY"]


class Meeting(gym.Env):

    """
        agents meet in a grid
    """

    def __init__(
        self,
        terminate_step=100,
        intermediate_r=False,
        obs_one_hot=False,
        agent_trans_noise=0.05,
    ):

        # env generic settings
        self.n_agent = get_n_agents()
        self.intermediate_reward = intermediate_r
        self.terminate_step = terminate_step

        # dimensions
        self.x_len = self.y_len = get_ct_dim()
        self.x_mean, self.y_mean = (
            np.mean(np.arange(self.x_len)),
            np.mean(np.arange(self.y_len)),
        )

        # probabilities
        self.agent_trans_noise = agent_trans_noise

        obs_size = self.x_len * self.y_len if obs_one_hot else 2

        self.obs_one_hot = obs_one_hot
        self.viewer = None
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-255, high=255, shape=(obs_size,), dtype=np.float32
        )
        self.state_space = spaces.Box(
            low=0, high=get_ct_dim(), shape=(self.n_agent * 2,), dtype=np.float32
        )

    def reset(self):
        self.step_n = 0

        # "game state" is really just the positions of all players
        self.agent_positions = np.stack(
            [self.rand_position() for _ in range(self.n_agent)]
        )

        obs = self.get_obs()

        return obs

    def get_state(self):
        return np.concatenate(self.agent_positions)

    def task_success(self):
        """ Whether all agents at the same location """
        return all(
            all(np.equal(a, self.agent_positions[0])) for a in self.agent_positions[1:]
        )

    def step(self, actions):
        self.step_n += 1
        assert len(actions) == self.n_agent

        self.agent_positions = self.move(
            self.agent_positions, actions, noise=self.agent_trans_noise
        )

        success = self.task_success()
        done = success or self.step_n >= self.terminate_step

        # if debug:
        #     print("Actions list:")
        #     print("Agent_0 \t action \t\t{}".format(ACTIONS[actions[0]]))
        #     print("Agent_1 \t action \t\t{}".format(ACTIONS[actions[1]]))

        return self.get_obs(), float(success), bool(done)

    def get_obs(self):
        if self.obs_one_hot:
            agt_pos_obs = self.one_hot_positions(self.agent_positions)
        else:
            agt_pos_obs = self.normalize_positions(self.agent_positions)

        # if debug:
        #     print("")
        #     print("Observations list:")
        #     for i in range(self.n_agent):
        #         print("Agent_" + str(i) +
        #               " \t self_loc  \t\t{}".format(self.agent_positions[i]))
        #         print("")

        return agt_pos_obs

    #################################################################################################
    # Helper functions

    def move(self, positions, directions, noise=0):
        translations = np.stack([self.translation(d, noise=noise) for d in directions])
        positions += translations
        return self.wrap_positions(positions)

    def rand_position(self):
        return np.array([randint(self.x_len), randint(self.y_len)])

    @staticmethod
    def translation(direction, noise=0.1):
        return TRANSLATION_TABLE[direction][
            np.random.choice(3, p=[noise / 2, 1 - noise, noise / 2])
        ]

    def flick(self, N, prob=0.3):
        mask = np.random.random(N.shape[0]).reshape(N.shape[0], -1) > prob
        if self.obs_one_hot:
            return N * mask
        else:
            flicker = np.stack([np.array([-1, -1]) for _ in range(N.shape[0])])
        return N * mask + flicker * np.logical_not(mask)

    def normalize_positions(self, positions):
        X, Y = np.split(positions, 2, axis=1)
        return np.concatenate([X / self.x_mean, Y / self.y_mean], axis=1)

    def one_hot_positions(self, positions):
        one_hot_vector = np.zeros((self.n_agent, self.x_len * self.y_len))
        index = positions[:, 1] * self.y_len + positions[:, 0]
        one_hot_vector[np.arange(self.n_agent), index] = 1
        return one_hot_vector

    def wrap_positions(self, positions):
        # fix translations which moved positions out of bound.
        X, Y = np.split(positions, 2, axis=1)
        return np.concatenate([X % self.x_len, Y % self.y_len], axis=1)


class Rescue(Meeting):
    """
    target location randomly initialized, penalized evey timestep, any agent step on target location ends episode
    """

    def __init__(
        self,
        terminate_step=100,
        intermediate_r=False,
        obs_one_hot=False,
        agent_trans_noise=0.05,
    ):
        super().__init__(terminate_step, intermediate_r, obs_one_hot, agent_trans_noise)
        # obs_size = self.x_len * self.y_len if obs_one_hot else 2
        # self.observation_space = spaces.Box(low=-255, high=255, shape=(obs_size*2,), dtype=np.float32)
        self.observation_space = self.state_space
        # self.state_space = spaces.Box(low=0, high=get_ct_dim(), shape=(self.n_agent * 2,), dtype=np.float32)

    def get_obs(self):
        # both agent observe the whole state
        o = np.concatenate(self.agent_positions)
        return np.stack([o, o])

    def reset(self):
        super().reset()
        self.target_position = self.rand_position()
        return self.get_obs()

    def task_success(self):
        """ Win condition """
        return any(all(np.equal(a, self.target_position)) for a in self.agent_positions)

    def step(self, a):
        _, r, t = super().step(a)
        return self.get_obs(), r - 1, t

# something requiring info gathering


