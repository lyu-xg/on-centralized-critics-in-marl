import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float32))
            self.ac_buffs.append(np.zeros((max_steps, adim), dtype=np.float32))
            self.rew_buffs.append(np.zeros(max_steps, dtype=np.float32))
            self.next_obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float32))
            self.done_buffs.append(np.zeros(max_steps, dtype=np.uint8))


        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[:, agent_i])
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=True)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)])

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]

class ReplayBufferEpi(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, nentries, max_episodes, epi_len, num_agents, obs_dims, ac_dims, s_dim):
        """
        Inputs:
            max_episodes (int): Maximum number of episodes to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_episodes = max_episodes
        self.epi_len = epi_len
        self.num_agents = num_agents

        # add buff to record state info
        self.state_buff = []
        self.next_state_buff = []
        self.obs_buffs = [] # n_agent x n_epi x epi_len x obsdim
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.valid_buffs = []

        self.state_buff.append(np.zeros((max_episodes, epi_len, s_dim), dtype=np.float32))
        self.next_state_buff.append(np.zeros((max_episodes, epi_len, s_dim), dtype=np.float32))
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_episodes, epi_len, odim), dtype=np.float32))
            self.ac_buffs.append(np.zeros((max_episodes, epi_len, adim), dtype=np.float32))
            self.rew_buffs.append(np.zeros((max_episodes, epi_len), dtype=np.float32))
            self.next_obs_buffs.append(np.zeros((max_episodes, epi_len, odim), dtype=np.float32))
            self.done_buffs.append(np.zeros((max_episodes, epi_len), dtype=np.uint8))
            self.valid_buffs.append(np.zeros((max_episodes, epi_len), dtype=np.uint8))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current epi index to write to (ovewrite oldest data)
        self.curr_exp_i = 0 # current exp index to write to (overite oldest data) 
        self.nentries = nentries  # handle multiple parallel environments

    def __len__(self):
        return self.filled_i

    def push(self, state, observations, actions, rewards, next_state, next_observations, dones, valids, all_epis_done=False):
        if self.curr_i + self.nentries > self.max_episodes:
            rollover = self.max_episodes - self.curr_i # num of indices to roll over

            self.state_buff[0] = np.roll(self.state_buff[0], rollover, axis=0)
            self.next_state_buff[0] = np.roll(self.next_state_buff[0], rollover, axis=0)
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover, axis=0)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover, axis=0)
                self.valid_buffs[agent_i] = np.roll(self.valid_buffs[agent_i],
                                                   rollover, axis=0)
            self.curr_i = 0
            self.filled_i = self.max_episodes
            self.flush()

        self.state_buff[0][self.curr_i:self.curr_i + self.nentries][:, self.curr_exp_i] \
                = np.vstack(state)
        self.next_state_buff[0][self.curr_i:self.curr_i + self.nentries][:, self.curr_exp_i] \
                = np.vstack(next_state)

        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + self.nentries][:,self.curr_exp_i] \
                    = np.vstack(observations[:, agent_i])
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + self.nentries][:,self.curr_exp_i] \
                    = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + self.nentries][:,self.curr_exp_i] \
                    = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + self.nentries][:,self.curr_exp_i] \
                    = np.vstack(next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + self.nentries][:,self.curr_exp_i] \
                    = dones[:, agent_i]
            self.valid_buffs[agent_i][self.curr_i:self.curr_i + self.nentries][:,self.curr_exp_i] \
                    = valids[:, agent_i]

        # move to next step for all epi
        self.curr_exp_i += 1
        if self.curr_exp_i >= self.epi_len or all_epis_done:
            self.curr_exp_i = 0
            self.curr_i += self.nentries
            if self.filled_i < self.max_episodes:
                self.filled_i += self.nentries
            if self.curr_i == self.max_episodes:
                self.curr_i = 0

            self.flush()

    def sample(self, N, to_gpu=False, norm_rews=False):
        if self.filled_i < self.max_episodes:
            inds = np.random.choice(np.arange(self.filled_i), size=N,
                                    replace=False)
        else:
            inds_0 = np.arange(0,self.curr_i)
            inds_1 = np.arange(self.curr_i+self.nentries, self.filled_i)
            inds = np.append(inds_0, inds_1)
            inds = np.random.choice(inds, size=N,
                                    replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [torch.from_numpy(self.rew_buffs[i][inds]).float() for i in range(self.num_agents)]

        return ([torch.from_numpy(self.state_buff[0][inds]).float() for _ in range(self.num_agents)],
                [torch.from_numpy(self.obs_buffs[i][inds]).float() for i in range(self.num_agents)],
                [torch.from_numpy(self.ac_buffs[i][inds]).float() for i in range(self.num_agents)],
                ret_rews,
                [torch.from_numpy(self.next_state_buff[0][inds]).float() for _ in range(self.num_agents)],
                [torch.from_numpy(self.next_obs_buffs[i][inds]).float() for i in range(self.num_agents)],
                [torch.from_numpy(self.done_buffs[i][inds]).float() for i in range(self.num_agents)],
                [torch.from_numpy(self.valid_buffs[i][inds]).float() for i in range(self.num_agents)])

    def get_average_rewards(self, N):
        if self.filled_i == self.max_episodes:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].sum(1).mean() for i in range(self.num_agents)]

    def flush(self):
        self.state_buff[0][self.curr_i:self.curr_i+self.nentries] = 0.0
        self.next_state_buff[0][self.curr_i:self.curr_i+self.nentries] = 0.0
        # refresh done_buff and valid_buff
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i+self.nentries] = 0.0
            self.ac_buffs[agent_i][self.curr_i:self.curr_i+self.nentries] = 0.0
            self.rew_buffs[agent_i][self.curr_i:self.curr_i+self.nentries] = 0.0
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i+self.nentries] = 0.0
            self.done_buffs[agent_i][self.curr_i:self.curr_i+self.nentries] = 0.0
            self.valid_buffs[agent_i][self.curr_i:self.curr_i+self.nentries] = 0.0
