# a dumb multi-agent environment in which each agent is solving its own cartpole problem
import gym
import numpy as np
from typing import List

from configs import get_n_agents, get_seed, get_env, get_ct_dim
from envs.matrix_game import GuessColor, TrickyStagHunt
from envs.dectiger import DecTiger
from envs.meeting import Meeting, Rescue
from envs.coke import SingleAgentCoke
from envs.utils import MultiAgentEnvLibWrapper
from envs.decpomdp import gymDecPOMDP, DPOMDP_ENVS
from marl_envs.particle_envs.make_env import make_env as make_particle_env
from marl_envs.my_env.cmotp import CMOTP1, CMOTP2, CMOTP3
from marl_envs.my_env.box_pushing import BoxPushing
from marl_envs.my_env.capture_target import CaptureTarget
from .shuo import EnvFindTreasure, EnvGoTogether, EnvCleaner, EnvMoveBox  # , BoxPushing


def BoxPushingModified(**kwargs):
    return BoxPushing(grid_dim=(4, 4), **kwargs)


class Env:
    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class Combiner(Env):
    """
    Combine single agent environment into one multi-agent one
    """

    def __init__(self, env_factory):
        self.envs = [env_factory() for _ in range(get_n_agents())]
        # self.envs = [gym.make('MountainCar-v0') for _ in range(n_agent)]
        self.spec = self.envs[0].spec
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.state_space = gym.spaces.Discrete(1)

        for env in self.envs:
            env.seed(get_seed())

    def reset(self) -> List[np.ndarray]:
        return [env.reset() for env in self.envs]

    def step(self, actions):
        state, rewards, dones, _ = zip(
            *[env.step(a) for a, env in zip(actions, self.envs)]
        )
        return state, rewards[0], any(dones)


particle_extensions = ("merge", "antipodal", "cross", "simple_coop_tag_v1")


class ParticleEnv(Env):
    def __init__(self, env_name):
        assert env_name in particle_extensions
        scene_name = "advanced_spread"
        self.horizen = 200
        if env_name == "merge":
            assert get_n_agents() == 2
        if env_name == "antipodal" or env_name == "cross":
            assert get_n_agents() == 4
        else:
            assert env_name == "simple_coop_tag_v1"
            scene_name = "simple_coop_tag_v1"
            self.horizen = 25
        obs_r = 0.6
        self.env = make_particle_env(
            scene_name,
            discrete_action_input=True,
            obs_r=obs_r,
            discrete_mul=2,
            config_name=env_name,
        )
        self.observation_space = self.env.observation_space[0]
        self.action_space = self.env.action_space[0]
        self.state_space = gym.spaces.Discrete(1)  # dummy placeholder

    def reset(self):
        self.t = 0
        return self.env.reset()

    def step(self, actions):
        self.t += 1
        _, s, r, d, _, _ = self.env.step(actions)
        if type(r) is list:
            r = float(r[0])
        return s, r, d[0] or self.t > self.horizen


def new_env() -> Env:
    env = get_env()

    if env == "capture_target":
        return MultiAgentEnvLibWrapper(
            CaptureTarget, grid_dim=(get_ct_dim(), get_ct_dim())
        )
    if env in particle_extensions:
        return ParticleEnv(env)
    if env == "guess_color":
        return GuessColor()
    if env == "cleaner":
        return EnvCleaner()
    if env == "find_treasure":
        return EnvFindTreasure()
    if env == "go_together":
        return EnvGoTogether()
    if env == "move_box":
        return EnvMoveBox()
    if env == "box_pushing":
        # return BoxPushing()
        return MultiAgentEnvLibWrapper(BoxPushingModified)
    if env == "stag":
        return TrickyStagHunt()
    if env == "dectiger":
        return DecTiger()
    if env == "meeting":
        return Meeting()
    if env == "rescue":
        return Rescue()
    if env == 'coke':
        return SingleAgentCoke()
    if env in DPOMDP_ENVS:
        return gymDecPOMDP(env)
    return Combiner(lambda: gym.make(env))
