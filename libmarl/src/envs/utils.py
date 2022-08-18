from gym import spaces
from warnings import warn
from numpy import array


class MultiAgentEnvLibWrapper:
    """
    Wrapper for my_envs in multiagent-envs
    """

    def __init__(self, env_class, horizen_limit=float("inf"), **kwargs):
        self.env = env_class(**kwargs)
        self.horizen_limit = horizen_limit
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render = self.env.render
        if hasattr(self.env, "state_space"):
            self.state_space = self.env.state_space
        else:
            warn("using dummy states for env in multiagent-envs")
            self.state_space = spaces.Discrete(1)

        if hasattr(self.env, "get_state"):
            self.get_state = self.env.get_state
        else:
            self.get_state = lambda: array([0])

    def reset(self):
        self.i_step = 0
        return self.env.reset()

    def step(self, actions):
        self.i_step += 1
        _, s, r, d, _, _ = self.env.step(actions)
        if type(r) is list:
            r = float(r[0])
        return s, r, d[0] or self.i_step > self.horizen_limit
