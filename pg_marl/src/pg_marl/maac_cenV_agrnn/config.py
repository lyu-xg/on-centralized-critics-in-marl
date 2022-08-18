from sacred import Experiment, observers
import IPython

ex = Experiment('partical_envs')
#ex.observers.append(observers.MongoObserver(url='10.200.205.239:27017', db_name='pg_marl'))

@ex.config
def default_config():
    env_name = 'pomdp_simple_speaker_listener'
    n_envs = 1
    total_epies = 60000
    max_epi_steps = 25

    gamma = 0.95
    a_lr = 1e-4
    c_lr = 1e-3
    c_train_iteration = 1

    # e-greedy decay params
    eps_start = 1.0
    eps_end = 0.01
    eps_decay_epis = 15000

    init_etrpy_w = 0.0
    end_etrpy_w = 0.0
    etrpy_w_l_d_epi = 10000 # entropy weith linear decay respects to epi

    train_freq = 2 # epidosic trainning
    c_target = False
    c_target_update_freq = 10
    c_target_soft_update = False
    tau = 0.01

    MC_baseline = False
    n_step_bootstrap = 0

    a_mlp_layer_size = 64
    a_rnn_layer_size = 64
    c_mlp_layer_size = 64

    grad_clip_value = None
    grad_clip_norm = 1.0

    discnt_a_loss = True

    eval_policy = False
    eval_freq = 100
    eval_num_epi = 10

    obs_last_action = False

    # particle env params
    prey_accel = 4.0
    prey_max_v = 1.3
    obs_r = 1.0
    obs_resolution = 3
    flick_p = 0.0
    enable_boundary = False
    benchmark = False
    discrete_mul = 1
    config_name = 'cross'

    # my_envs params
    grid_dim = [4, 4]
    target_rand_move = False
    n_target = 1
    # box pushing
    random_init = False
    small_box_only = False
    terminal_reward_only = False
    big_box_reward = 100
    small_box_reward = 10
    n_agent = 2
    # cmotp
    cmotp_version = 1
    cmotp_local_obs_shape = [3, 3] 

    n_run = 1
    run_idx = 0
    save_rate = 1000
    save_dir = "trial"

    save_ckpt=False
    save_ckpt_time=23
    resume = False

@ex.named_config
def MC_baseline():
    MC_baseline = True

@ex.named_config
def c_target():
    c_target = True

@ex.named_config
def c_target_soft_update():
    c_target_soft_update = True

@ex.named_config
def no_discnt_a_loss():
    discnt_a_loss = False

@ex.named_config
def eval_policy():
    eval_policy = True

@ex.named_config
def enable_boundary():
    enable_boundary = True

@ex.named_config
def obs_last_action():
    obs_last_action = True

@ex.named_config
def benchmark():
    benchmark = True

@ex.named_config
def target_rand_move():
    target_rand_move = True

@ex.named_config
def random_init():
    random_init = True

@ex.named_config
def small_box_only():
    small_box_only = True

@ex.named_config
def terminal_reward_only():
    terminal_reward_only = True

@ex.named_config
def save_ckpt():
    save_ckpt = True

@ex.named_config
def resume():
    resume = True
