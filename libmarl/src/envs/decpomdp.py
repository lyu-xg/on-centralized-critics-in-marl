import os
import numpy as np
from gym import spaces, Env
from configs import get_n_agents
from itertools import product
from collections import defaultdict
from rl_parsers.dpomdp import parse

DPOMDP_ENVS = {
    'grid_small': 'dpomdp/GridSmall.dpomdp',
    'grid': 'dpomdp/grid3x3corners.dpomdp',
    'recycling': 'dpomdp/recycling.dpomdp',
    'mars': 'dpomdp/Mars.dpomdp',
    'boxpushing': 'dpomdp/boxpushing.dpomdp',
    'firefighting': 'dpomdp/fireFighting.dpomdp',
    'firefighting_4house': 'dpomdp/fireFighting_2_4_3.dpomdp',
    'wireless': 'dpomdp/wirelessWithOverhead.dpomdp',
    'long_fire_fight': 'dpomdp/longFireFight.dpomdp',
    'dtiger': 'dpomdp/dectiger.dpomdp'
}

class gymDecPOMDP(Env):
    def __init__(self, problem, episode_limit=100):
        self.episode_limit = episode_limit
        filename = os.path.join(os.path.dirname(__file__), DPOMDP_ENVS.get(problem))
        if not filename:
            raise FileNotFoundError(problem + 'environment not found')
        with open(filename) as f:
            self.d = parse(f.read())

        self.n_agents = len(self.d.agents)
        assert self.n_agents == get_n_agents()

        self.n_states = len(self.d.states)
        self.states = tuple(range(self.n_states))

        self.n_obs = len(self.d.observations[0])
        self.observations = tuple(range(self.n_obs))
        assert all(self.d.observations[0] == O for O in self.d.observations)

        self.n_actions = len(self.d.actions[0])
        self.actions = tuple(range(self.n_actions))
        assert all(self.d.actions[0] == A for A in self.d.actions)

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_obs)
        self.state_space = spaces.Discrete(self.n_states)

        self.joint_observations = tuple(product(self.observations, repeat=self.n_agents))
        self.joint_actions = tuple(product(self.actions, repeat=self.n_agents))

    
    @property
    def no_obs(self):
        return [np.zeros(self.n_obs) for _ in self.d.agents]
    
    def reset(self):
        self.i_step = 1
        self.state = np.random.choice(self.states, p=self.d.start)
        return self.no_obs
    
    def step(self, a):
        self.i_step += 1
        a = tuple(a)
        probs = [self.d.T[(*a, self.state, sp)] for sp in self.states]
        new_state = np.random.choice(self.states, p=probs)
        obs = self._emit(a, new_state)
        reward = float(self.d.R[(*a, self.state, new_state, *obs)])
        is_done = bool(self.d.reset[(*a, new_state)])
        self.state = new_state
        return self.onehot_joint_obs(obs), reward, is_done or self.i_step > self.episode_limit

    def get_state(self):
        return self.onehot_state(self.state)
        
    def _emit(self, actions, new_state):
        probs = [self.d.O[(*actions, new_state, *o)] for o in self.joint_observations]
        i =  np.random.choice(len(self.joint_observations), p=probs)
        return self.joint_observations[i]
    
    def onehot_state(self, s):
        res = np.zeros(self.n_states)
        res[s] = 1
        return res
    
    def onehot_joint_obs(self, O):
        return [self.onehot_obs(o) for o in O]
    
    def onehot_obs(self, o):
        res = np.zeros(self.n_obs)
        res[o] = 1
        return res
    
