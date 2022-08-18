import numpy as np
import torch

from collections import deque

class Memory:

    def __init__(self, env, obs_last_action=False, size=1):
        self.buf = deque(maxlen=size)
        self.n_agent = env.n_agent

        self.ZERO_STATE = torch.zeros(env.get_env_info()['state_shape']).view(1,-1)
        if not obs_last_action:
            self.ZERO_OBS = [torch.zeros(dim).view(1,-1) for dim in env.obs_size]
        else:
            self.ZERO_OBS = [torch.zeros(o_dim+a_dim).view(1,-1) for o_dim, a_dim in zip(*[env.obs_size, env.n_action])]
        self.ZERO_ACT = [torch.tensor(0).view(1,-1)] * self.n_agent
        self.ZERO_REWARD = [torch.tensor(0.0).view(1,-1)] * self.n_agent
        self.ONE_TERMINATE = [torch.tensor(1.0).view(1,-1)] * self.n_agent
        self.ZERO_DISCOUNT = [torch.tensor(0.0).view(1,-1)] * self.n_agent
        self.ZERO_EXPV = [torch.tensor(0.0).view(1,-1)]*self.n_agent

        self.ZERO_PADDING = [(self.ZERO_STATE,
                              self.ZERO_OBS,
                              self.ZERO_ACT,
                              self.ZERO_REWARD,
                              self.ZERO_STATE,
                              self.ZERO_OBS,
                              self.ONE_TERMINATE,
                              self.ZERO_DISCOUNT,
                              self.ZERO_EXPV)]

    def append(self, transition):
        self.scenario_cache.append(transition)

    def flush_buf_cache(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
    
    def _scenario_cache_reset(self):
        raise NotImplementedError

class Memory_epi(Memory):
    
    def __init__(self, *args, **kwargs):
        super(Memory_epi, self).__init__(*args, **kwargs)
        self._scenario_cache_reset()

    def flush_buf_cache(self):
        self.buf.append(self.scenario_cache)
        self._scenario_cache_reset()
    
    def sample(self):
        batch = list(self.buf)
        return self._padding_batches(batch)

    def _scenario_cache_reset(self):
        self.scenario_cache = []

    def _padding_batches(self, batch):
        epi_len = [len(epi) for epi in batch] 
        max_len = max(epi_len)
        batch = [epi + self.ZERO_PADDING * (max_len - len(epi)) for epi in batch]
        return batch, max_len, epi_len
