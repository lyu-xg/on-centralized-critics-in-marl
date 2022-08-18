import os
from sacred import Experiment, observers
import wandb

def is_local():
    return os.uname().nodename == 'mbp.local'

project_name = "test" if is_local() else 'debug'

ex = Experiment(name=project_name)


@ex.config
def default_config():
    # seed = 42
    project_name = "debug"
    method = "ppo"
    discount_factor = 0.99
    learning_rate = 3e-4
    learning_rate_for_critic = 3e-3
    n_agents = 2
    total_rollouts = 5000
    copy_target_net_interval = 1  # update target network(s) every n rollout
    batch_size = 64
    epochs = 3
    clip_param = 0.2
    buffer_size = 800

    number_of_eval_episodes = 5
    eval_discount = False

    entropy_scale = 0.01
    clip_grad_norm = 0.5

    network = [
        {"recurrency": False, "h_size": 128},
        {"recurrency": True, "h_size": 64},
    ]

    critic_type = 'history'

    trace_len = 3

    # all agents' policies share the same parameter: homogeneous agents
    is_share_param = False

    # training paramaters
    episodic_train = False

    common_feature = False

    normalize_reward = False

    # curiosity default params
    icm = False
    intrinsic_reward_integration = 0.01
    reward_scale = 0.01
    icm_forward_loss_weight = 0.2

    # coordination params
    is_centralized_critic = False

    # environment
    # env = 'CartPole-v1'
    # env = 'merge'
    env = "guess_color"

    # env = 'capture_target' # specific configs
    ct_dim = 10  # 4x4 grid
    target_flick_prob = 0.0

    obs_r = 0.6

    rand_walk = False
    smoothing = 0.02  # amount of each ep_reward is incorporated into running_reward
    reward_window = 0
    episodic_reward_record = False

    use_predefined_value_func = False

    use_expected_sarsa = False

    ppo_value_clip = False

    use_marginalized_q = False

    use_both_value_loss = False

    
@ex.config_hook
def wand_init(config, command_name, logger):
    name = 'debug' if is_local() else config['project_name']
    wandb.init(project=name, config=config)
    return config


@ex.named_config
def cartpole():
    env = "CartPole-v1"


@ex.named_config
def antipodal():
    env = "antipodal"
    n_agents = 4


@ex.named_config
def cross():
    env = "cross"
    n_agents = 4


@ex.named_config
def merge():
    env = "merge"
    n_agents = 2


@ex.named_config
def spread():
    learning_rate = 0.0001
    learning_rate_for_critic = 0.001
    batch_size = 128
    trace_len = 1


@ex.named_config
def tag():
    env = "simple_coop_tag_v1"
    n_agents = 3
    total_rollouts = 20000
    batch_size = 500 
    discount_factor = 0.95
    trace_len = 1


@ex.named_config
def ppo():
    method = "ppo"


@ex.named_config
def rand_walk():
    rand_walk = True


@ex.named_config
def icm():
    icm = True


@ex.named_config
def lstm():
    network = network = [
        {"recurrency": False, "h_size": 64},
        {"recurrency": True, "h_size": 64},
    ]
    trace_len = 3


@ex.named_config
def ac():
    method = "ac"


@ex.named_config
def a2c():
    method = "a2c"


@ex.named_config
def reinforce():
    method = "reinforce"


@ex.named_config
def two_agents():
    n_agents = 2


@ex.named_config
def central_critic():
    is_centralized_critic = True


@ex.named_config
def state_based_critic():
    is_centralized_critic = True
    critic_type = 'state'

@ex.named_config
def state_history_based_critic():
    is_centralized_critic = True
    critic_type = 'state+history'


@ex.named_config
def ep_train():
    episodic_train = True


@ex.named_config
def dectiger():
    env = 'dectiger'
    trace_len = 2
    learning_rate = 0.0003
    # tuning shows slow critic works better, but that contradicts convergence assumption for actor critic method
    # it might due to the fact that the learning rate needed to be annealed for general convergence
    learning_rate_for_critic = 0.00003
    entropy_scale = 0.001
    total_rollouts = 1500
    batch_size = 256

@ex.named_config
def coke():
    env = 'coke'
    n_agents = 1
    learning_rate = 0.008
    learning_rate_for_critic = 0.008
    entropy_scale = 0.01

@ex.named_config
def true_value():
    use_predefined_value_func = True
    env = "guess_color"


@ex.named_config
def shuo_setup():
    reward_window = 50
    episodic_reward_record = True
    learning_rate = 3e-4
    learning_rate_for_critic = 1e-3
    discount_factor = 0.98
    n_agents = 2
    entropy_scale = 0.3


@ex.named_config
def shuo_cleaner():
    env = "cleaner"
    total_rollouts = 60000
    episodic_train = True
    network = [
        {"recurrency": False, "h_size": 128},
        {"recurrency": False, "h_size": 128},
    ]


@ex.named_config
def shuo_find_treasure():
    env = "find_treasure"
    total_rollouts = 10000
    network = [
        {"recurrency": False, "h_size": 128},
        {"recurrency": False, "h_size": 128},
    ]
    episodic_train = True


@ex.named_config
def shuo_go_together():
    env = "go_together"
    total_rollouts = 10000
    buffer_size = 2000
    network = [
        {"recurrency": False, "h_size": 128},
        {"recurrency": False, "h_size": 128},
    ]


@ex.named_config
def shuo_move_box():
    env = "move_box"
    total_rollouts = 20000
    episodic_train = True
    network = [
        {"recurrency": False, "h_size": 256},
        {"recurrency": False, "h_size": 256},
    ]


@ex.named_config
def shuo_box_pushing():
    env = "box_pushing"
    total_rollouts = 15000
    episodic_train = True
    network = [
        {"recurrency": False, "h_size": 256},
        {"recurrency": False, "h_size": 256},
    ]


@ex.named_config
def true_value_stag():
    env = "stag"
    use_predefined_value_func = True
    buffer_size = 2000
    entropy_scale = 0.0


####################################################################
# getter functions


@ex.capture
def get_method(method):
    return method


@ex.capture
def get_batch_size(batch_size):
    return batch_size


@ex.capture
def get_ppo_params(buffer_size, batch_size, epochs, clip_param):
    return buffer_size, batch_size, epochs, clip_param


@ex.capture
def get_buffer_size(buffer_size):
    return buffer_size


@ex.capture
def get_n_agents(n_agents: int):
    return n_agents


@ex.capture
def get_entropy_scale(entropy_scale: float):
    return entropy_scale


@ex.capture
def get_clip_grad_norm(clip_grad_norm: float):
    return clip_grad_norm


@ex.capture
def get_episodic_train(episodic_train):
    return episodic_train


@ex.capture
def get_icm_params(intrinsic_reward_integration, reward_scale, icm_forward_loss_weight):
    return intrinsic_reward_integration, reward_scale, icm_forward_loss_weight


@ex.capture
def get_icm_flag(icm):
    return icm


@ex.capture
def get_normalization_reward(normalize_reward):
    return normalize_reward


@ex.capture
def get_learning_rate(learning_rate):
    return learning_rate


@ex.capture
def get_learning_rate_for_critic(learning_rate_for_critic):
    return learning_rate_for_critic


@ex.capture
def get_discount_factor(discount_factor):
    return discount_factor


@ex.capture
def get_is_centralized_critic(is_centralized_critic):
    return is_centralized_critic


@ex.capture
def get_common_feature(common_feature):
    return common_feature


@ex.capture
def get_seed(seed):
    return seed


@ex.capture
def get_network(network):
    return network


@ex.capture
def get_env(env):
    return env


@ex.capture
def get_ct_dim(ct_dim):
    return ct_dim


@ex.capture
def get_rand_walk(rand_walk):
    return rand_walk


@ex.capture
def get_smoothing(smoothing):
    return smoothing


@ex.capture
def get_target_flick_prob(target_flick_prob):
    return target_flick_prob


@ex.capture
def is_reward_func_aligned(env):
    return env not in ("merge", "antipodal", "cross", "guess_color")


@ex.capture
def get_epochs(epochs):
    return epochs


@ex.capture
def get_is_q_func(method):
    return method in ("qac",)


@ex.capture
def get_use_predefined_value_func(use_predefined_value_func):
    return use_predefined_value_func


@ex.capture
def get_reward_window(reward_window):
    return reward_window


@ex.capture
def get_critic_learning_ratio(learning_rate, learning_rate_for_critic):
    return learning_rate_for_critic / learning_rate


@ex.capture
def get_trace_len(trace_len):
    return trace_len


@ex.capture
def get_is_share_param(is_share_param):
    return is_share_param


@ex.capture
def get_critic_type(critic_type):
    assert critic_type in ('history', 'state', 'state+history')
    return critic_type


@ex.capture
def get_use_expected_sarsa(use_expected_sarsa):
    return use_expected_sarsa


@ex.capture
def get_ppo_value_clip(ppo_value_clip):
    return ppo_value_clip


@ex.capture
def get_copy_target_net_interval(copy_target_net_interval):
    return copy_target_net_interval


@ex.capture
def get_number_of_eval_episodes(number_of_eval_episodes):
    return number_of_eval_episodes


@ex.capture
def get_eval_discount(eval_discount):
    return eval_discount


@ex.capture
def get_use_marginalized_q(use_marginalized_q):
    return use_marginalized_q


@ex.capture
def get_is_q_func(method):
    return method in ("qac", "dppo")


@ex.capture
def get_is_dueling(method):
    return method in ("dppo",)


@ex.capture
def get_use_both_value_loss(use_both_value_loss):
    assert get_is_dueling()
    return use_both_value_loss

@ex.capture
def get_is_state_based_critic(critic_type):
    return critic_type in ('state', 'state+history')

@ex.capture
def get_is_history_based_critic(critic_type):
    return critic_type in ('history', 'state+history')