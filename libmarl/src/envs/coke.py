import numpy as np
from numpy.random import randint
from gym import spaces, Env
from configs import get_n_agents

ASK_PENALTY = -3.0
CORRECT_REWARD = 10.0
INCORRECT_REWARD = -10.0


class SingleAgentCoke:
    def __init__(self):
        assert get_n_agents() == 1
        self.actions = ("ask", "serve_coke", "serve_diet_coke")
        self.observations = ("no_obs", "want_coke", "want_diet_coke")
        self.states = ("want_coke", "want_diet_coke")
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(len(self.observations))
        self.state_space = spaces.Discrete(len(self.states))

    def reset(self):
        self.state = randint(len(self.states))
        return self.encode_obs(0)

    def get_state(self):
        return self.one_hot(self.state, len(self.states))
        # return np.zeros(len(self.states))

    def step(self, a):
        a = a[0]
        if a == 0:  # ask
            return self.encode_obs(self.state+1), ASK_PENALTY, False
        else:
            r = CORRECT_REWARD if self.state == a-1 else INCORRECT_REWARD
            return self.encode_obs(0), r, True

    def one_hot(self, i, l):
        res = np.zeros(l)
        res[i] = 1
        return res

    def encode_obs(self, obs):
        one_hot = self.one_hot(obs, len(self.observations))
        return np.array([one_hot])

