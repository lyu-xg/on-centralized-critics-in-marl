import numpy as np
from gym import spaces, Env
from configs import get_n_agents

"""
Dec-tiger domain from http://masplan.org/problem_domains

2 agents

Uniform random initial state: { tiger-left, tiger-right }

Actions for each agent: { listen, open-left, open-right }

Observations: { hear-left, hear-right }

Observation model: 
    both hear correct door: 0.7225 
    one hear correct door: 0.1275
    both hear wrong door: 0.0225

Reward model: 
    both listen: -2
    both open wrong door: -50
    both open right door: +20
    agent open different door: -100
    one agent open wrong door + other agent listen: -101
    one agent open right door + other agent listen: +9

"""


ACTIONS = ("open-left", "open-right", "listen")
STATES = ("tiger-left", "tiger-right")
OBSERVAIONS = ("hear-left", "hear-right")
JOINT_OBSERVATIONS_RIGHT = ((1, 1), (1, 0), (0, 1), (0, 0))
JOINT_OBSERVATIONS_LEFT = ((0, 0), (1, 0), (0, 1), (1, 1))


class DecTiger(Env):
    def __init__(self):
        super().__init__()
        self.n_agent = 2
        assert get_n_agents() == 2
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Discrete(2)
        self.state_space = spaces.Discrete(1)

    def reset(self):
        self.tiger = np.random.randint(0, 2)
        return self.obs_out(None)

    @staticmethod
    def obs_out(obs):
        # onehot encoded observations, supports empty observation
        res = np.zeros((2, 2))
        if obs is not None:
            for i, o in enumerate(obs):
                res[(i, o)] = 1
        return res

    def get_state(self):
        # state space only got two states: tiger-left (0) and riget-right (1)
        return [self.tiger]

    def step(self, a):
        done = True
        o = None
        a1, a2 = a
        if a1 == a2 == 2:  # both listen
            obs = (
                JOINT_OBSERVATIONS_LEFT if self.tiger == 0 else JOINT_OBSERVATIONS_RIGHT
            )
            i = np.random.choice(4, p=(0.7225, 0.1275, 0.1275, 0.0225))
            o = obs[i]
            r = -2
            done = False
        elif a1 == a2 != self.tiger:
            r = 20  # both opened treasure door
        elif a1 == a2 == self.tiger:
            r = -50  # both opened tiger door
        elif a1 != a2:
            if 2 in a:  # one agent listens
                if self.tiger in a:
                    r = -101  # one agent opened tiger door
                else:
                    r = 9  # one agent opened treasure door
            else:
                r = -100  # different doors was open
        return self.obs_out(o), float(r), done
