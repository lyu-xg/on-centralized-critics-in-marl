import argparse
import torch
import os
import numpy as np
import random
import pickle
import time
import wandb

from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer, ReplayBufferEpi
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC

from marl_envs.my_env.capture_target import CaptureTarget as CT
from marl_envs.my_env.small_box_pushing import SmallBoxPushing as SBP
from marl_envs.my_env.dectiger import DecTiger
from marl_envs.particle_envs.make_env import make_env


def make_parallel_env(config, env_args, n_rollout_threads, seed):
    env_id = config.env_id

    def get_env_fn(rank):
        def init_env():
            if env_id.startswith('CT'):
                env = CT(grid_dim=tuple(config.grid_dim), **env_args)
            elif env_id.startswith('SBP'):
                env = SBP(tuple(config.grid_dim), **env_args)
            elif env_id == 'dectiger':
                env = DecTiger()
            else:
                env = make_env(env_id, 
                               discrete_action=True, 
                               discrete_action_input=True, 
                               **env_args)
                if seed is not None:
                    env.seed(seed + rank)
            if seed is not None:
                np.random.seed(seed + rank)
                random.seed(seed + rank)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def make_test_env(config, env_args, seed):
    env_id = config.env_id
    def get_env_fn(rank):
        def init_env():
            if env_id.startswith('CT'):
                env = CT(grid_dim=tuple(config.grid_dim), **env_args)
            elif env_id.startswith('SBP'):
                env = SBP(tuple(config.grid_dim), **env_args)
            elif env_id == 'dectiger':
                env = DecTiger()
            else:
                env = make_env(env_id, 
                               discrete_action=True,
                               discrete_action_input=True,
                               **env_args)
                env.seed(seed)
            return env
        return init_env
    return DummyVecEnv([get_env_fn(0)])

def run(config):
    name = 'debug' if os.uname().nodename == 'lxg-mbp.local' else config.env_id
    c = vars(config)
    c['critic_type'] = 'SC' if c['state_critic'] else 'OC'
    wandb.init(project='maac_'+name, config=c)
    if config.logger:
        model_dir = Path('./models') / config.env_id / config.model_name
        if not model_dir.exists():
            run_num = 1
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                             model_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                run_num = 1
            else:
                run_num = max(exst_run_nums) + 1

        curr_run = 'run%i' % run_num
        run_dir = model_dir / curr_run
        log_dir = run_dir / 'logs'
        os.makedirs(log_dir)
        logger = SummaryWriter(str(log_dir))
    else:
        logger = None

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.set_num_threads(1)

    if config.env_id.startswith('CT'):
        env_args = {'terminate_step': config.episode_length,
                     'n_target': config.n_target,
                     'n_agent': config.n_agent}
    elif config.env_id.startswith('BP') or config.env_id.startswith('SBP'):
        env_args = {'terminate_step': config.episode_length,
                    'random_init': config.random_init,
                    'small_box_only': config.small_box_only,
                    'terminal_reward_only': config.terminal_reward_only,
                    'big_box_reward': config.big_box_reward,
                    'small_box_reward': config.small_box_reward,
                    'n_agent': config.n_agent}
    else:
        env_args = {'max_epi_steps': config.episode_length,
                    'prey_accel': config.n_target,
                    'prey_max_v': config.prey_max_v,
                    'obs_r': config.obs_r,
                    'obs_resolution': config.obs_resolution,
                    'flick_p': config.flick_p,
                    'enable_boundary': config.enable_boundary,
                    'benchmark': config.benchmark,
                    'discrete_mul': config.discrete_mul,
                    'config_name': config.config_name}

    env = make_parallel_env(config, env_args, config.n_rollout_threads, config.seed)

    # create an env for testing
    env_test = make_test_env(config, env_args, config.seed)

    env_info = env_test.envs[0].get_env_info()

    model = AttentionSAC.init_from_env(env,
                                       env_info,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale,
                                       grad_clip_norm=config.grad_clip_norm,
                                       state_critic=config.state_critic)

    replay_buffer = ReplayBufferEpi(config.n_rollout_threads,
                                    config.buffer_length, 
                                    config.episode_length,
                                    model.nagents,
                                    [obsp.shape[0] for obsp in env.observation_space],
                                    [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                    for acsp in env.action_space],
                                    env_info['state_shape'])

    time_start = time.time()
    time_counter = time.time() 
    if config.resume:
        t, ep_start, test_returns = load_ckpt(config.run_idx, replay_buffer, model, config.save_dir)
    else:
        t = 0
        ep_start = 0
        test_returns = []

    for ep_i in range(ep_start, config.n_episodes, config.n_rollout_threads):

        if ep_i % (config.eval_freq - (config.eval_freq % config.n_rollout_threads)) == 0:
            test_return = evaluate(env_test, 
                                   model, 
                                   config.gamma, 
                                   config.episode_length, 
                                   eval_num_epi=config.eval_num_epi)
            time_cost = round((time.time() - time_counter) / 3600,3)
            time_counter = time.time()
            print(f"{[config.run_idx]} Finished: {ep_i}/{config.n_episodes} took {time_cost}hr, Evaluate learned policies with averaged returns {test_return/config.n_agent} ...", flush=True)
            test_returns.append(test_return)
            wandb.log({'test_return': test_return/config.n_agent})

        # print("Episodes %i-%i of %i" % (ep_i + 1,
        #                                 ep_i + 1 + config.n_rollout_threads,
        #                                 config.n_episodes))

        state, obs = env.reset()
        torch_H = [[None] for _ in range(obs.shape[1])]
        model.prep_rollouts(device='cpu')

        dones_count = []
        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False).unsqueeze(1)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            torch_agent_actions, torch_H = model.step(torch_obs, H=torch_H, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]

            # return extra state info
            next_state, next_obs, rewards, dones, _  = env.step(actions)

            valids = np.ones_like(dones)

            # modify valids value depending on terminated env 
            if len(dones_count) > 0:
                valids[np.array(dones_count)] = False

            # count the terminated env
            if dones.any():
                dones_epis = list(np.nonzero(dones.all(axis=1))[0])
                for epi_id in dones_epis:
                    if epi_id not in dones_count:
                        dones_count.append(epi_id)

            # if all envs done, update recording index in the buffer
            if len(dones_count) == config.n_rollout_threads:
                replay_buffer.push(state, obs, agent_actions, rewards, next_state, next_obs, dones, valids, all_epis_done=True)
                break
            else:
                replay_buffer.push(state, obs, agent_actions, rewards, next_state, next_obs, dones, valids)
 
            obs = next_obs
            state = next_state

            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)

                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')

        if ep_i % (config.save_rate - (config.save_rate % config.n_rollout_threads)) == 0:
            save_test_data(config.run_idx, test_returns, config.save_dir)

        if (time.time() - time_start) // 3600 >= 23:
            save_ckpt(config.run_idx, t, ep_i+config.n_rollout_threads, test_returns, replay_buffer, model, config.save_dir)
            print('save')
            break

        if logger is not None:
            ep_rews = replay_buffer.get_average_rewards(
                config.n_rollout_threads)
            for a_i, a_ep_rew in enumerate(ep_rews):
                logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                                  a_ep_rew * config.episode_length, ep_i)

        # if ep_i % config.save_interval < config.n_rollout_threads:
        #     model.prep_rollouts(device='cpu')
        #     os.makedirs(run_dir / 'incremental', exist_ok=True)
        #     model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
        #     model.save(run_dir / 'model.pt')

    # model.save(run_dir / 'model.pt')
    save_test_data(config.run_idx, test_returns, config.save_dir)
    save_ckpt(config.run_idx, t, ep_i+config.n_rollout_threads, test_returns, replay_buffer, model, config.save_dir)
    env.close()
    env_test.close()
    print("Finish entire training ... ", flush=True)
    if config.logger:
        logger.export_scalars_to_json(str(log_dir / 'summary.json'))
        logger.close()

def evaluate(env, model, gamma, episode_length, eval_num_epi=10):
    for agent in model.agents:
        agent.policy.eval()
    R = 0.0
    for ep_i in range(eval_num_epi):
        obs = env.reset()
        torch_H = [[None] for _ in range(obs.shape[1])]
        dones = np.array([[False, False]])

        for et_i in range(episode_length):
            if dones.any():
                break
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False).unsqueeze(1)
                         for i in range(model.nagents)]
            torch_agent_actions, torch_H = model.step(torch_obs, H=torch_H, explore=True)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = [[ac[i] for ac in agent_actions] for i in range(1)]
            next_obs, rewards, dones, _  = env.step(actions)
            obs = next_obs
            R += gamma**et_i*np.sum(rewards)

    for agent in model.agents:
        agent.policy.train()

    return R/eval_num_epi

def save_test_data(run_idx, data, save_dir):
    # with open("./performance/" + save_dir + "/test/test_perform" + str(run_idx) + ".pickle", 'wb') as handle:
    #     pickle.dump(data, handle)
    return

def save_ckpt(run_idx, steps, ep_i, test_returns, replay_buffer, model, save_dir, max_save=2):

    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_generic_" + "{}.tar"
    for n in list(range(max_save-1, 0, -1)):
        os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
    PATH = PATH.format(1)

    torch.save({'t': steps,
                'ep_i': ep_i,
                'replay_buffer_state': replay_buffer.state_buff,
                'replay_buffer_next_state': replay_buffer.next_state_buff,
                'replay_buffer_obs': replay_buffer.obs_buffs,
                'replay_buffer_ac': replay_buffer.ac_buffs,
                'replay_buffer_rew': replay_buffer.rew_buffs,
                'replay_buffer_next_obs': replay_buffer.next_obs_buffs,
                'replay_buffer_done': replay_buffer.done_buffs,
                'replay_buffer_valid': replay_buffer.valid_buffs,
                'filled_i': replay_buffer.filled_i,
                'curr_i': replay_buffer.curr_i,
                'random_state': random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_random_state': torch.random.get_rng_state(),
                'test_returns': test_returns},
                PATH)

    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_critic_" + "{}.tar"
    for n in list(range(max_save-1, 0, -1)):
        os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
    PATH = PATH.format(1)

    torch.save({'critic_net_state_dict': model.critic.state_dict(),
                'critic_tgt_net_state_dict': model.target_critic.state_dict(),
                'critic_optimizer_state_dict': model.critic_optimizer.state_dict()}, 
               PATH)

    for idx, agent in enumerate(model.agents):
        PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_agent_" + str(idx) + "{}.tar"

        for n in list(range(max_save-1, 0, -1)):
            os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
        PATH = PATH.format(1)
        torch.save({'actor_net_state_dict': agent.policy.state_dict(),
                    'actor_tgt_net_state_dict': agent.target_policy.state_dict(),
                    'actor_optimizer_state_dict': agent.policy_optimizer.state_dict()},
                   PATH)

def load_ckpt(run_idx, replay_buffer, model, save_dir):
    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_generic_" + "1.tar"
    ckpt = torch.load(PATH)
    ep_i = ckpt['ep_i']
    t = ckpt['t']
    test_returns = ckpt['test_returns']
    random.setstate(ckpt['random_state'])
    np.random.set_state(ckpt['np_random_state'])
    torch.set_rng_state(ckpt['torch_random_state'])

    replay_buffer.curr_i = ckpt['curr_i']
    replay_buffer.filled_i = ckpt['filled_i']
    replay_buffer.state_buff = ckpt['replay_buffer_state']
    replay_buffer.next_state_buff = ckpt['replay_buffer_next_state']
    replay_buffer.obs_buffs = ckpt['replay_buffer_obs']
    replay_buffer.ac_buffs = ckpt['replay_buffer_ac']
    replay_buffer.rew_buffs = ckpt['replay_buffer_rew']
    replay_buffer.next_obs_buffs = ckpt['replay_buffer_next_obs']
    replay_buffer.done_buffs = ckpt['replay_buffer_done']
    replay_buffer.valid_buffs = ckpt['replay_buffer_valid']
    replay_buffer.flush()

    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_critic_" + "1.tar"
    ckpt = torch.load(PATH)
    model.critic.load_state_dict(ckpt['critic_net_state_dict'])
    model.target_critic.load_state_dict(ckpt['critic_tgt_net_state_dict'])
    model.critic_optimizer.load_state_dict(ckpt['critic_optimizer_state_dict'])

    for idx, agent in enumerate(model.agents):
        PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_agent_" + str(idx) + "1.tar"
        ckpt = torch.load(PATH)
        agent.policy.load_state_dict(ckpt['actor_net_state_dict'])
        agent.target_policy.load_state_dict(ckpt['actor_tgt_net_state_dict'])
        agent.policy_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])

    return t, ep_i, test_returns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default='simple_collect_treasure', type=str,
                        help="Name of environment")
    parser.add_argument("--model_name", default='test', type=str,
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--buffer_length", default=int(1e4), type=int)
    parser.add_argument("--n_episodes", default=50000, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=10, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--state_critic", action='store_true')
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.0005, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--grad_clip_norm", default=0.5, type=float)
    # env args
    parser.add_argument("--n_agent", default=8, type=int)
    parser.add_argument('--grid_dim', nargs=2, default=[4,4], type=int)
    parser.add_argument("--n_target", default=1, type=int)
    parser.add_argument("--small_box_only", action='store_true')
    parser.add_argument("--terminal_reward_only", action='store_true')
    parser.add_argument("--random_init", action='store_true')
    parser.add_argument("--small_box_reward", default=100.0, type=float)
    parser.add_argument("--big_box_reward", default=100.0, type=float)
    # particle envs args
    parser.add_argument("--prey_accel", default=4.0, type=float)
    parser.add_argument("--prey_max_v", default=1.3, type=float)
    parser.add_argument("--flick_p", default=0.0, type=float)
    parser.add_argument("--obs_r", default=2.0, type=float)
    parser.add_argument("--enable_boundary", action='store_true')
    parser.add_argument("--discrete_mul", default=1, type=int)
    parser.add_argument("--config_name", default="antipodal", type=str)
    parser.add_argument("--benchmark", action='store_true')
    parser.add_argument("--obs_resolution", default=8, type=int)
    # evaluation
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--eval_num_epi", default=10, type=int)
    parser.add_argument("--run_idx", default=0, type=int)
    parser.add_argument("--save_dir", default='test', type=str)
    parser.add_argument("--save_rate", default=1000, type=int)
    parser.add_argument("--logger", action='store_true')
    parser.add_argument("--resume", action='store_true')

    config = parser.parse_args()

    # create the dirs to save results
    os.makedirs("./performance/" + config.save_dir + "/test", exist_ok=True)
    os.makedirs("./performance/" + config.save_dir + "/ckpt", exist_ok=True)

    run(config)
