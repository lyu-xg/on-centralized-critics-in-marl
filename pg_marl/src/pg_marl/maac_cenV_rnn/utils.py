
class Agent:

    def __init__(self):
        self.idx = None
        self.actor_net = None
        self.actor_optimizer = None
        self.actor_loss = None

        self.critic_net = None
        self.critic_tgt_net = None
        self.critic_optimizer = None
        self.critic_loss = None

class Linear_Decay(object):

    def __init__ (self, total_steps, init_value, end_value):
        self.total_steps = total_steps
        self.init_value = init_value
        self.end_value = end_value

    def get_value(self, step):
        frac = min(float(step) / self.total_steps, 1.0)
        return self.init_value + frac * (self.end_value-self.init_value)
