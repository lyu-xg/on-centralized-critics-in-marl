import numpy as np
from numpy.random import randint
from gym import spaces, Env
from configs import get_n_agents

ACTIONS = (1, 2, 3)
OBSERVATIONS = (1, 2)
TERMINAL_STATE = np.array([[1], [1]])


def single_agent_reward(s, a):
    # `s` is the state of the other agent
    if a == 2:
        return 5.0
    if s == a:
        return 10.0
    else:
        return -10.0


class GuessColor(Env):
    def __init__(self):
        super().__init__()
        self.n_agent = 2
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Discrete(1)
        assert get_n_agents() == 2

    def rand_state(self):
        self.state = np.random.randint(0, 2, self.n_agent)

    def reset(self):
        self.rand_state()
        return self.state.reshape((-1, 1))

    @staticmethod
    def true_q(s, a):
        # input s is a joint observation
        # input a is a joint action
        # returns the true q value of s
        return np.mean(GuessColor._reward(s, a))

    @staticmethod
    def true_decen_q(s, a):
        if a == 2:
            return 5.0
        return 0.0

    @staticmethod
    def _reward(state, action):
        return [
            single_agent_reward(state[1], action[0]),
            single_agent_reward(state[0], action[1]),
        ]

    def step(self, actions):
        R = self._reward(self.state, actions)
        return TERMINAL_STATE, R, True


class TrickyStagHunt(Env):
    def __init__(self):
        super().__init__()
        self.n_agent = 2
        assert get_n_agents() == 2
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(4)

    def initialize(self):
        self.state = (0, 0)
        self.done = False

    def is_init_state(self):
        return self.state == (0, 0)

    def rand_state2(self):
        self.state = (randint(1, 3), randint(1, 3))

    def reset(self):
        self.initialize()
        return self.state_out(self.state)

    @staticmethod
    def one_hot(s):
        res = np.zeros(4)
        res[s] = 1
        return res

    @staticmethod
    def one_hot_to_int(s):
        return int(sum(np.arange(4) * s)[0])

    @staticmethod
    def state_out(S):
        # return np.array(S).reshape((-1, 1))
        return np.vstack(TrickyStagHunt.one_hot(s) for s in S)

    def terminate(self):
        self.done = True
        self.state = (3, 3)

    def step1(self, U):
        if any(u == 0 for u in U):  # (0,0) (1,0) and (0,1) are the safe option here
            self.terminate()
            return 0.5
        # trainsition into the tricky state
        self.rand_state2()
        return 0.0

    def step2(self, U):
        # assuming we are in the tricky mode
        # return 1 if reversed(S) = U
        (s1, s2), (u1, u2) = self.state, U
        self.terminate()
        return float(s1 == u2 + 1 and s2 == u1 + 1)
        # reward = 0.0
        # if s1 == u2: reward += 1
        # else: reward -= 1
        # if s2 == u1: reward += 1
        # else: reward -= 1
        # return reward

    def step(self, u):
        if self.is_init_state():
            r = self.step1(u)
        else:
            r = self.step2(u)
        return self.state_out(self.state), r, self.done

    """
        Perhaps we need to tweak the reward function to make learning smoother
        Or change the way in which the agents enter the tricky states.
    """

    @staticmethod
    def true_q(S, A):
        # true joint Q
        S = list(map(TrickyStagHunt.one_hot_to_int, np.vsplit(S, 2)))
        if S == [3, 3]:
            return 0.0  # terminal state
        if S == [0, 0]:
            if any(a == 0 for a in A):
                return 1.0
            return 0.5
        else:  # tricky states
            if all(s == a for s, a in zip(S, reversed(A))):
                return 1.0
            return 0.0

    @staticmethod
    def true_decen_q(s, a):
        if s[3]:
            return 0.0
        if s[0]:
            if a == 0:  # (0,0) or (0,1)
                return 0.5
            return 0.375  # (0,1) or (1,1), assume uniform policy from ther other agent we get 0.375
        else:  # trickly states
            return 0.25
