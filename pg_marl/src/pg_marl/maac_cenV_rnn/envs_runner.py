import numpy as np
import random
import torch
import torch.nn.functional as F

from multiprocessing import Process, Pipe

def worker(child, env, gamma, seed):
    """
    Worker function which interacts with the environment over remote
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        while True:
            # wait cmd sent by parent
            cmd, data = child.recv()
            if cmd == 'step':
                action, obs, reward, terminate, valid, _  = env.step(data)

                # sent experience back
                child.send((last_obs, action, reward, obs, terminate, gamma**step))

                last_obs = obs
                R += gamma**step * sum(reward)
                #R += sum(reward)
                step += 1
            
            elif cmd == 'get_return':
                child.send(R)

            elif cmd == 'reset':
                last_obs =  env.reset() # List[array]
                h_state = [None] * env.n_agent
                last_action = [-1] * env.n_agent
                step = 0
                R = 0.0

                child.send((last_obs, h_state, last_action))
            elif cmd == 'close':
                child.close()
                break
            elif cmd == 'get_rand_states':
                rand_states = {'random_state': random.getstate(),
                               'np_random_state': np.random.get_state()}
                child.send(rand_states)
            elif cmd == 'load_rand_states':
                random.setstate(data['random_state'])
                np.random.set_state(data['np_random_state'])
            else:
                raise NotImplementerError
 
    except KeyboardInterrupt:
        print('EnvRunner worker: caught keyboard interrupt')
    except Exception as e:
        print('EnvRunner worker: uncaught worker exception')
        raise

class EnvsRunner(object):
    """
    Environment runner which runs mulitpl environemnts in parallel in subprocesses
    and communicates with them via pipe
    """

    def __init__(self, ex, env, n_envs, controller, memory, max_epi_steps, gamma, seed, obs_last_action=False):
        
        self.ex = ex
        self.env = env
        self.max_epi_steps = max_epi_steps
        self.n_envs = n_envs
        self.n_agent = env.n_agent
        # controllers for getting next action via current actor nn
        self.controller = controller
        # create connections via Pipe
        self.parents, self.children = [list(i) for i in zip(*[Pipe() for _ in range(n_envs)])]
        # create multip processor with multiple envs
        self.envs = [Process(target=worker, args=(child, env, gamma, seed+idx)) for idx, child in enumerate(self.children)]
        # replay buffer
        self.memory = memory
        # observe last actions
        self.obs_last_action = obs_last_action

        self.episodes = [[] for i in range(n_envs)]
        self.returns = []
        self.smoothed_return = 0.0

        # trigger each processor
        for env in self.envs:
            env.daemon = True
            env.start()

        for child in self.children:
            child.close()

    def run(self, eps, n_epis=1):

        self._reset()

        while self.n_epi_count < n_epis:
            self._step(eps)

    def close(self):
        [parent.send(('close', None)) for parent in self.parents]
        [parent.close() for parent in self.parents]
        [env.terminate() for env in self.envs]
        [env.join() for env in self.envs]

    def _step(self, eps):

        for idx, parent in enumerate(self.parents):
            
            actions, self.h_states[idx] = self.controller.select_action(self.last_obses[idx], self.h_states[idx], eps=eps) # List[Int]

            # send cmd to trigger env step
            parent.send(("step", actions))
            self.step_count[idx] += 1

        # collect envs' returns
        for idx, parent in enumerate(self.parents):
            # env_return is (last_obs, a, r, obs, t)
            env_return = parent.recv()
            env_return = self._exp_to_tensor(idx, env_return)
            # env_return become (last_obs, a, r, obs, t, discnt, exp_valid)

            self.episodes[idx].append(env_return)
            self.last_obses[idx] = env_return[3]
            if self.obs_last_action:
                self.last_actions[idx] = env_return[1]

            # if episode is done, add it to memory buffer
            if env_return[-3][0] or self.step_count[idx] == self.max_epi_steps:
                self.n_epi_count += 1
                self.memory.scenario_cache += self.episodes[idx]
                self.memory.flush_buf_cache()

                # collect the return
                parent.send(("get_return", None))
                R = parent.recv()
                self.returns.append(R)
                self.smoothed_return = 0.1 * R + (1 - 0.1) * self.smoothed_return
                self.ex.log_scalar('epi_return', R)
                self.ex.log_scalar('smoothed_return', self.smoothed_return)

                # when episode is done, immediately start a new one
                parent.send(("reset", None))
                self.last_obses[idx], self.h_states[idx], self.last_actions[idx] = parent.recv()
                self.last_obses[idx] = self.obs_to_tensor(self.last_obses[idx])
                if self.obs_last_action:
                    self.last_actions[idx] = self.action_to_tensor(self.last_actions[idx])
                    self.last_obses[idx] = self.rebuild_obs(self.env, self.last_obses[idx], self.last_actions[idx])
                self.episodes[idx] = []
                self.step_count[idx] = 0

    def _reset(self):
        # send cmd to reset envs
        for parent in self.parents:
            parent.send(("reset", None))

        self.last_obses, self.h_states, self.last_actions = [list(i) for i in zip(*[parent.recv() for parent in self.parents])]
        self.last_obses = [self.obs_to_tensor(obs) for obs in self.last_obses] #List[List[tensor]]
        if self.obs_last_action:
            self.last_actions = [self.action_to_tensor(a) for a in self.last_actions]
            # reconstruct obs to observe actions
            self.last_obses = [self.rebuild_obs(self.env, obs, a) for obs, a in zip(*[self.last_obses, self.last_actions])]
        self.n_epi_count = 0
        self.step_count = [0] * self.n_envs
        self.episodes = [[] for i in range(self.n_envs)]

    def _exp_to_tensor(self, env_idx, exp):
        # exp (last_obs, a, r, obs, t, discnt)
        last_obs = [torch.from_numpy(o).float().view(1,-1) for o in exp[0]]
        a = [torch.tensor(a).view(1,-1) for a in exp[1]]
        r = [torch.tensor(r).float().view(1,-1) for r in exp[2]]
        obs = [torch.from_numpy(o).float().view(1,-1) for o in exp[3]]
        t = [torch.tensor(t).float().view(1,-1) for t in exp[4]]
        disct = [torch.tensor(exp[5]).float().view(1,-1)] * self.n_agent
        exp_v = [torch.tensor([1.0]).view(1,-1)] * self.n_agent
        # re-construct obs if obs last action
        if self.obs_last_action:
            last_obs = self.rebuild_obs(self.env, last_obs, self.last_actions[env_idx])
            obs = self.rebuild_obs(self.env, obs, a)
        return (last_obs, a, r, obs, t, disct, exp_v)

    @staticmethod
    def obs_to_tensor(obs):
        return [torch.from_numpy(o).float().view(1,-1) for o in obs]

    @staticmethod
    def action_to_tensor(action):
        return [torch.tensor(a).view(1,-1) for a in action]

    @staticmethod
    def rebuild_obs(env, obs, action):
        new_obs = []
        for o, a, a_dim in zip(*[obs, action, env.n_action]):
            if a == -1:
                one_hot_a = torch.zeros(a_dim).view(1,-1)
            else:
                one_hot_a = F.one_hot(a.view(-1), a_dim).float()
            new_obs.append(torch.cat([o, one_hot_a], dim=1))
        return new_obs
