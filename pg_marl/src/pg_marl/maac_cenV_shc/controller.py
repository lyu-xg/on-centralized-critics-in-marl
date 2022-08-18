import torch 

from torch.distributions import Categorical
from .envs_runner import EnvsRunner
from .models import Actor
from .utils import Agent

class MAC(object):
    
    def __init__(self, env, obs_last_action, a_mlp_layer_size=64, a_rnn_layer_size=64):
        self.env = env
        self.n_agent = env.n_agent
        self.obs_last_action = obs_last_action

        self.a_mlp_layer_size = a_mlp_layer_size
        self.a_rnn_layer_size = a_rnn_layer_size

        self.eval_returns = []

        self._build_agent()

    def select_action(self, obses, h_states, eps=0.0, test_mode=False):
        actions = [] # List[Int]
        new_h_states = []
        with torch.no_grad():
            for idx, agent in enumerate(self.agents):
                action_logits, new_h_state = agent.actor_net(obses[idx].view(1,1,-1), h_states[idx], eps=eps, test_mode=test_mode)
                action_prob = Categorical(logits=action_logits[0])
                action = action_prob.sample().item()
                actions.append(action)
                new_h_states.append(new_h_state)
        return actions, new_h_states

    def evaluate(self, gamma, max_epi_steps, n_episode=10):
        R = 0.0

        for _ in range(n_episode):
            t = 0
            step = 0
            last_obs = EnvsRunner.obs_to_tensor(self.env.reset())
            if self.obs_last_action:
                last_action = EnvsRunner.action_to_tensor([-1]*self.env.n_agent)
                last_obs = EnvsRunner.rebuild_obs(self.env, last_obs, last_action)
            h_state = [None]*self.n_agent

            while not t and step < max_epi_steps:

                a, h_state = self.select_action(last_obs, h_state, test_mode=True)
                a, obs, r, t, v, _ = self.env.step(a)
                last_obs = EnvsRunner.obs_to_tensor(obs)
                if self.obs_last_action:
                    a = EnvsRunner.action_to_tensor(a)
                    last_obs = EnvsRunner.rebuild_obs(self.env, last_obs, a)
                R += gamma**step*sum(r)
                step += 1
                t = all(t)

        self.eval_returns.append(R/n_episode)
        # print(f"Evaluate learned policies with averaged return {self.eval_returns[-1]}")

    def _build_agent(self):
        self.agents = []
        for idx in range(self.n_agent):
            agent = Agent()
            agent.idx = idx
            agent.actor_net = Actor(self._get_input_shape(idx), self.env.n_action[idx], self.a_mlp_layer_size, self.a_rnn_layer_size)
            self.agents.append(agent)

    def _get_input_shape(self, agent_idx):
        if not self.obs_last_action:
            return self.env.obs_size[agent_idx]
        else:
            return self.env.obs_size[agent_idx] + self.env.n_action[agent_idx]
