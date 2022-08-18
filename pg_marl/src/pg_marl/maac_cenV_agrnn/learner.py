import torch
import copy
import numpy as np

from torch.optim import Adam
from torch.nn.utils import clip_grad_value_, clip_grad_norm_

from itertools import chain
from .models import Critic
from .utils import Linear_Decay

class Learner(object):
    
    def __init__(self, env, controller, memory, gamma=0.95, a_lr=1e-2, c_lr=1e-2, 
                 c_train_iteration=1, init_etrpy_w=0.0, end_etrpy_w=0.0, etrpy_w_l_d_epi=10000, 
                 c_target=False, c_target_update_freq=50, tau=0.01,
                 grad_clip_value=None, grad_clip_norm=None,
                 MC_baseline=False, n_step_bootstrap=0, 
                 c_mlp_layer_size=64, discnt_a_loss=True):

        self.env = env
        self.n_agent = env.n_agent
        self.agents_action_space = env.n_action
        self.controller = controller
        self.memory = memory

        self.gamma = gamma
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.c_train_iteration = c_train_iteration

        self.etrpy_w_decay_sys = Linear_Decay(etrpy_w_l_d_epi, init_etrpy_w, end_etrpy_w)

        self.c_mlp_layer_size = c_mlp_layer_size
        # if true, using target net
        self.c_target = c_target
        self.c_target_update_freq = c_target_update_freq
        self.tau = tau # used for target-net soft update

        self.grad_clip_value = grad_clip_value
        self.grad_clip_norm = grad_clip_norm

        self.MC_baseline = MC_baseline
        self.n_step_bootstrap = n_step_bootstrap

        self.discnt_a_loss=discnt_a_loss

        self._create_joint_critic()
        self._set_optimizer()

    def train(self, epi_done, eps):

        batch, trace_len, epi_len = self.memory.sample()
        batch_size = len(batch)

        ############################# train centralized critic ###################################
        cen_batch  = self._cat_joint_exps(batch)

        state, action, reward, n_state, terminate, discnt, exp_v = zip(*cen_batch)

        state = torch.cat(state).view(batch_size, trace_len, -1)
        action = torch.cat(action).view(batch_size, trace_len, -1)
        reward = torch.cat(reward).view(batch_size, trace_len, -1)
        n_state = torch.cat(n_state).view(batch_size, trace_len, -1)
        terminate = torch.cat(terminate).view(batch_size, trace_len, -1)
        discnt = torch.cat(discnt).view(batch_size, trace_len, -1)
        exp_v = torch.cat(exp_v).view(batch_size, trace_len, -1)

        ##############################  calculate critic loss and optimize the critic_net ####################################
        for _ in range(self.c_train_iteration):
            if not self.MC_baseline:
                # NOTE WE SHOULD NOT BACKPROPAGATE CRITIC_NET BY N_STATE
                if self.c_target:
                    Gt = self._get_bootstrap_return(reward, n_state, terminate, epi_len, self.joint_critic_tgt_net)
                else:
                    Gt = self._get_bootstrap_return(reward, n_state, terminate, epi_len, self.joint_critic_net)
            else:
                if self.c_target:
                    Gt = self._get_discounted_return(reward, n_state, terminate, epi_len, self.joint_critic_tgt_net)
                else:
                    # G_t = r_t + gamma^1*r_t+1 + ....+ gamma^{T-t}*r_T with last step bootstrap
                    Gt = self._get_discounted_return(reward, n_state, terminate, epi_len, self.joint_critic_net)

            TD = Gt - self.joint_critic_net(state)
            self.joint_critic_loss = torch.sum(exp_v * TD * TD) / exp_v.sum()
            self.joint_critic_optimizer.zero_grad()
            self.joint_critic_loss.backward()
            if self.grad_clip_value is not None:
                clip_grad_value_(self.joint_critic_net.parameters(), self.grad_clip_value)
            if self.grad_clip_norm is not None:
                clip_grad_norm_(self.joint_critic_net.parameters(), self.grad_clip_norm)
            self.joint_critic_optimizer.step()

        ##############################  calculate actor loss using the updated critic ####################################
        if not self.MC_baseline:
            Gt = self._get_bootstrap_return(reward, n_state, terminate, epi_len, self.joint_critic_net)
        else:
            Gt = self._get_discounted_return(reward, n_state, terminate, epi_len, self.joint_critic_net)
        V_value = self.joint_critic_net(state).detach()

        dec_batch = self._sep_joint_exps(batch)

        for agent, batch in zip(self.controller.agents, dec_batch):
            
            state, action, reward, n_state, terminate, discnt, exp_v = zip(*batch)

            state = torch.cat(state).view(batch_size, trace_len, -1)
            action = torch.cat(action).view(batch_size, trace_len, -1)
            reward = torch.cat(reward).view(batch_size, trace_len, -1)
            n_state = torch.cat(n_state).view(batch_size, trace_len, -1)
            terminate = torch.cat(terminate).view(batch_size, trace_len, -1)
            discnt = torch.cat(discnt).view(batch_size, trace_len, -1)
            exp_v = torch.cat(exp_v).view(batch_size, trace_len, -1)

            # advantage value
            adv_value = Gt - V_value
            action_logits = agent.actor_net(state, eps=eps)[0]
            # log_pi(a|s) 
            log_pi_a = action_logits.gather(-1, action)
            # H(pi(.|s)) used as exploration bonus
            pi_entropy = torch.distributions.Categorical(logits=action_logits).entropy().view(batch_size, trace_len, 1)
            # actor loss
            alpha = self.etrpy_w_decay_sys.get_value(epi_done)
            if self.discnt_a_loss:
                actor_loss = torch.sum(exp_v * discnt * (log_pi_a * adv_value + alpha * pi_entropy), dim=1)
            else:
                actor_loss = torch.sum(exp_v * (log_pi_a * adv_value + alpha * pi_entropy), dim=1)
            # NOTE THE CORRECT LOSS SHOULD BE AVERAGE OVER EPISODES
            agent.actor_loss = -1 * torch.sum(actor_loss) / exp_v.sum()

        ############################# optimize each actor-net ########################################

        for agent in self.controller.agents:
            agent.actor_optimizer.zero_grad()
            agent.actor_loss.backward()
            if self.grad_clip_value is not None:
                clip_grad_value_(agent.actor_net.parameters(), self.grad_clip_value)
            if self.grad_clip_norm is not None:
                clip_grad_norm_(agent.actor_net.parameters(), self.grad_clip_norm)
            agent.actor_optimizer.step()

    def update_critic_target_net(self, soft=False):
        if not soft:
            self.joint_critic_tgt_net.load_state_dict(self.joint_critic_net.state_dict())
        else:
            with torch.no_grad():
                for q, q_targ in zip(self.joint_critic_net.parameters(), self.joint_critic_tgt_net.parameters()):
                    q_targ.data.mul_(1 - self.tau)
                    q_targ.data.add_(self.tau * q.data)

    def _create_joint_critic(self):
        input_dim = self._get_input_shape()
        self.joint_critic_net = Critic(input_dim, 1, self.c_mlp_layer_size)
        self.joint_critic_tgt_net = Critic(input_dim, 1, self.c_mlp_layer_size)
        self.joint_critic_tgt_net.load_state_dict(self.joint_critic_net.state_dict())

    def _get_input_shape(self):
        return self.env.get_env_info()['state_shape']

    def _set_optimizer(self):
        for agent in self.controller.agents:
            agent.actor_optimizer = Adam(agent.actor_net.parameters(), lr=self.a_lr)
        self.joint_critic_optimizer = Adam(self.joint_critic_net.parameters(), lr=self.c_lr)

    def _cat_joint_exps(self, joint_exps):
        # process the sampled batch for training a centralized critic
        # the centralized critic only receives ground trueth state
        exp = []
        for s, o, a, r, s_n, o_n, t, discnt, exp_v in chain(*joint_exps):
            exp.append([s,
                        torch.tensor(np.ravel_multi_index(a, self.agents_action_space)).view(1,-1),
                        r[0], 
                        s_n,
                        t[0],
                        discnt[0],
                        exp_v[0]])
        return exp

    def _sep_joint_exps(self, joint_exps):
        # seperate the joint experience for individual agents
        exps = [[] for _ in range(self.n_agent)]
        for s, o, a, r, s_n, o_n, t, discnt, exp_v in chain(*joint_exps):
            for i in range(self.n_agent):
                exps[i].append([o[i], a[i], r[i], o_n[i], t[i], discnt[i], exp_v[i]])
        return exps

    def _get_discounted_return(self, reward, n_state, terminate, epi_len, critic_net):
        Gt = copy.deepcopy(reward)
        for epi_idx, epi_r in enumerate(Gt):
            end_step_idx = epi_len[epi_idx]-1
            if not terminate[epi_idx][end_step_idx]:
                # implement last step bootstrap
                epi_r[end_step_idx] += self.gamma * critic_net(n_state[epi_idx][end_step_idx]).detach()
            for idx in range(end_step_idx-1, -1, -1):
                epi_r[idx] = epi_r[idx] + self.gamma * epi_r[idx+1]
        return Gt

    def _get_bootstrap_return(self, reward, n_state, terminate, epi_len, critic_net):
        if self.n_step_bootstrap:
            # implement n-step bootstrap
            bootstrap = critic_net(n_state).detach()
            Gt = copy.deepcopy(reward)
            for epi_idx, epi_r in enumerate(Gt):
                end_step_idx = epi_len[epi_idx]-1
                if not terminate[epi_idx][end_step_idx]:
                    epi_r[end_step_idx] += self.gamma * bootstrap[epi_idx][end_step_idx]
                for idx in range(end_step_idx-1, -1, -1):
                    if idx > end_step_idx-self.n_step_bootstrap:
                        epi_r[idx] = epi_r[idx] + self.gamma * epi_r[idx+1]
                    else:
                        epi_r[idx] = self._get_n_step_discounted_bootstrap_return(reward[epi_idx][idx:idx+self.n_step_bootstrap], bootstrap[epi_idx][idx+self.n_step_bootstrap-1])
        else:
            Gt = reward + self.gamma * critic_net(n_state).detach() * (-terminate + 1)
        return Gt

    def _get_n_step_discounted_bootstrap_return(self, reward, bootstrap):
        discount = torch.pow(torch.ones(1, 1) * self.gamma, torch.arange(self.n_step_bootstrap)).view(self.n_step_bootstrap, 1)
        Gt = torch.sum(discount * reward) + self.gamma**self.n_step_bootstrap * bootstrap
        return Gt
