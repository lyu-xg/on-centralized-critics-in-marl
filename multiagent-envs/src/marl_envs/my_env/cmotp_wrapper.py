import numpy as np

from .cmotp import CMOTP

class CMOTP_WRAPPER(CMOTP):

    def __init__(self, state_image=False, *args, **kwargs):
        super(CMOTP_WRAPPER, self).__init__(*args, **kwargs)
        self.state_image = state_image

    def step(self, actions):
        actions, observations, rewards, terminates, valids, stats = super().step(actions)
        if not self.state_image:
            return actions, [o.reshape(-1) / 255.0 for o in observations], rewards, terminates, valids, stats
        else:
            return actions, observations, rewards, terminates, valids, stats

    def reset(self):
        return super().reset()

    def get_state(self):
        if self.state_image:
            return self.s_t
        else:
            agents_xy = np.concatenate(np.split(np.stack([np.array(self.agents_x), np.array(self.agents_y)]).T, 2), axis=1)
            goods_xy = np.array([[self.goods_x, self.goods_y]])
            state = np.concatenate([agents_xy, goods_xy], axis=1)
        return state.reshape(-1) / (self.c.DIM[0]-1)

    def get_env_info(self):
        return {'state_shape': 6,
                'obs_shape': self.obs_size,
                'n_actions': self.n_action,
                'n_agents': self.n_agent,
                'episode_limit': self.terminate_step}

    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)

    @property
    def obs_size(self):
        return [np.prod(self.obs_shape)] * self.n_agent

    @property
    def action_spaces(self):
        return [self.action_space] * self.n_agent

    @property
    def n_action(self):
        return [self.c.ACTIONS] * self.n_agent



