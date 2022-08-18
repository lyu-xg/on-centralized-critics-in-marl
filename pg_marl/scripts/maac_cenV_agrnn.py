import gym
import torch
import random
import numpy as np
import os
import pickle
import time

from marl_envs.particle_envs.make_env import make_env

from pg_marl.maac_cenV_agrnn.config import ex
from pg_marl.maac_cenV_agrnn.envs_runner import EnvsRunner
from pg_marl.maac_cenV_agrnn.memory import Memory_epi
from pg_marl.maac_cenV_agrnn.learner import Learner
from pg_marl.maac_cenV_agrnn.controller import MAC
from pg_marl.maac_cenV_agrnn.utils import Linear_Decay

@ex.main
def main(env_name, n_envs, total_epies, max_epi_steps, gamma, a_lr, c_lr, 
         c_train_iteration, eps_start, eps_end, eps_decay_epis, init_etrpy_w, end_etrpy_w, etrpy_w_l_d_epi, 
         train_freq, c_target, c_target_update_freq, c_target_soft_update, tau,
         MC_baseline, n_step_bootstrap, grad_clip_value, grad_clip_norm,
         a_mlp_layer_size, c_mlp_layer_size, a_rnn_layer_size,
         discnt_a_loss, eval_policy, eval_freq, eval_num_epi, obs_last_action, prey_accel, prey_max_v, 
         obs_r, obs_resolution, flick_p, enable_boundary, benchmark, discrete_mul, config_name, 
         grid_dim, target_rand_move, n_target,
         random_init, small_box_only, terminal_reward_only, big_box_reward, small_box_reward, n_agent,
         cmotp_version, cmotp_local_obs_shape,
         seed, n_run, run_idx, save_rate, save_dir,
         save_ckpt, save_ckpt_time, resume):

    for idx in range(n_run):

        if n_run > 1:
            run_idx = idx
            seed = idx * 10 + 1

        # set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)

        # create the dirs to save results
        os.makedirs("./performance/" + save_dir + "/train", exist_ok=True)
        os.makedirs("./performance/" + save_dir + "/test", exist_ok=True)
        os.makedirs("./performance/" + save_dir + "/ckpt", exist_ok=True)

        # collect params
        actor_params = {'a_mlp_layer_size': a_mlp_layer_size,
                        'a_rnn_layer_size': a_rnn_layer_size}

        critic_params = {'c_mlp_layer_size': c_mlp_layer_size}

        hyper_params = {'gamma': gamma,
                        'a_lr': a_lr,
                        'c_lr': c_lr,
                        'c_train_iteration': c_train_iteration,
                        'init_etrpy_w': init_etrpy_w,
                        'end_etrpy_w': end_etrpy_w,
                        'etrpy_w_l_d_epi': etrpy_w_l_d_epi,
                        'c_target': c_target,
                        'c_target_update_freq': c_target_update_freq,
                        'tau': tau,
                        'grad_clip_value': grad_clip_value,
                        'grad_clip_norm': grad_clip_norm,
                        'MC_baseline': MC_baseline,
                        'n_step_bootstrap': n_step_bootstrap,
                        'discnt_a_loss': discnt_a_loss}

        particle_env_params = {'max_epi_steps': max_epi_steps,
                      'prey_accel': prey_accel,
                      'prey_max_v': prey_max_v,
                      'obs_r': obs_r,
                      'obs_resolution': obs_resolution,
                      'flick_p': flick_p,
                      'enable_boundary': enable_boundary,
                      'benchmark': benchmark,
                      'discrete_mul': discrete_mul,
                      'config_name': config_name}
        
        ct_params = {'terminate_step': max_epi_steps,
                     'n_target': n_target,
                     'n_agent': n_agent}

        box_pushing_params = {'terminate_step': max_epi_steps,
                              'random_init': random_init,
                              'small_box_only': small_box_only,
                              'terminal_reward_only': terminal_reward_only,
                              'big_box_reward': big_box_reward,
                              'small_box_reward': small_box_reward,
                              'n_agent': n_agent}

        cmotp_params = {'version': cmotp_version,
                        'local_obs_shape': cmotp_local_obs_shape}

        # create env
        env = make_env(env_name, discrete_action_input=True, **particle_env_params)
        env.seed(seed)

        # create buffer
        memory = Memory_epi(env, obs_last_action, size=train_freq)
        # cretate controller
        controller = MAC(env, obs_last_action, **actor_params) 
        # create parallel envs runner
        envs_runner = EnvsRunner(ex, env, n_envs, controller, memory, max_epi_steps, gamma, seed, obs_last_action)
        # create learner
        learner = Learner(env, controller, memory, **hyper_params, **critic_params)
        # create epsilon calculator for implementing e-greedy exploration policy
        eps_call = Linear_Decay(eps_decay_epis, eps_start, eps_end)

        epi_count = 0
        t_ckpt = time.time()
        if resume:
            epi_count = load_checkpoint(run_idx, save_dir, controller, learner, envs_runner)

        while epi_count < total_epies:

            if eval_policy and epi_count % (eval_freq - (eval_freq % train_freq)) == 0:
                controller.evaluate(gamma, max_epi_steps, eval_num_epi)
                print(f"{[run_idx]} Finished: {epi_count}/{total_epies} Evaluate learned policies with averaged returns {controller.eval_returns[-1]} ...", flush=True)

            # update eps
            eps = eps_call.get_value(epi_count)
            # let envs run a certain number of episodes accourding to train_freq
            envs_runner.run(eps, n_epis=train_freq)

            # perform a2c learning
            learner.train(epi_count, eps)

            epi_count += train_freq

            if c_target:
                if c_target_soft_update:
                    learner.update_critic_target_net(soft=True)
                elif epi_count % c_target_update_freq == 0:
                    learner.update_critic_target_net()
            
            if epi_count % save_rate == 0:
                save_train_data(run_idx, envs_runner.returns, save_dir)
                save_test_data(run_idx, controller.eval_returns, save_dir)

            # if epi_count % 20 == 0:
            #     print(f"{[run_idx]} Finished: {epi_count}/{total_epies}  Latest Episodic Return: {envs_runner.returns[-1]:.2f}  Smoothed Return: {envs_runner.smoothed_return:.2f}", flush=True)

            if save_ckpt and (time.time() - t_ckpt) / 3600 >= save_ckpt_time:
                save_checkpoint(run_idx, epi_count, controller, learner, envs_runner, save_dir)
                t_ckpt = time.time()
                break

        save_train_data(run_idx, envs_runner.returns, save_dir)
        save_test_data(run_idx, controller.eval_returns, save_dir)
        save_checkpoint(run_idx, epi_count, controller, learner, envs_runner, save_dir)
        envs_runner.close()

    print("Finish entire training ... ", flush=True)

def save_train_data(run_idx, data, save_dir):
    with open("./performance/" + save_dir + "/train/train_perform" + str(run_idx) + ".pickle", 'wb') as handle:
        pickle.dump(data, handle)

def save_test_data(run_idx, data, save_dir):
    with open("./performance/" + save_dir + "/test/test_perform" + str(run_idx) + ".pickle", 'wb') as handle:
        pickle.dump(data, handle)

def save_checkpoint(run_idx, epi_count, controller, learner, envs_runner, save_dir, max_save=2):

    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_genric_" + "{}.tar"

    for n in list(range(max_save-1, 0, -1)):
        os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
    PATH = PATH.format(1)

    torch.save({
                'epi_count': epi_count,
                'random_state': random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_random_state': torch.random.get_rng_state(),
                'envs_runner_returns': envs_runner.returns,
                'controller_eval_returns': controller.eval_returns,
                'smoothed_return': envs_runner.smoothed_return,
                'joint_critic_net_state_dict': learner.joint_critic_net.state_dict(),
                'joint_critic_tgt_net_state_dict': learner.joint_critic_tgt_net.state_dict(),
                'joint_critic_optimizer_state_dict': learner.joint_critic_optimizer.state_dict()
                }, PATH)

    for idx, parent in enumerate(envs_runner.parents):
        parent.send(('get_rand_states', None))

    for idx, parent in enumerate(envs_runner.parents):
        PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_env_rand_states_" + str(idx) + "{}.tar"
        rand_states = parent.recv()

        for n in list(range(max_save-1, 0, -1)):
            os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
        PATH = PATH.format(1)

        torch.save(rand_states, PATH)

    for idx, agent in enumerate(controller.agents):
        PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_agent_" + str(idx) + "{}.tar"

        for n in list(range(max_save-1, 0, -1)):
            os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
        PATH = PATH.format(1)

        torch.save({
                    'actor_net_state_dict': agent.actor_net.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    },PATH)

def load_checkpoint(run_idx, save_dir, controller, learner, envs_runner):

    # load generic stuff
    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_genric_" + "1.tar"
    ckpt = torch.load(PATH)
    epi_count = ckpt['epi_count']
    random.setstate(ckpt['random_state'])
    np.random.set_state(ckpt['np_random_state'])
    torch.set_rng_state(ckpt['torch_random_state'])
    envs_runner.returns = ckpt['envs_runner_returns']
    controller.eval_returns = ckpt['controller_eval_returns']
    envs_runner.smoothed_return = ckpt['smoothed_return']
    learner.joint_critic_net.load_state_dict(ckpt['joint_critic_net_state_dict'])
    learner.joint_critic_tgt_net.load_state_dict(ckpt['joint_critic_tgt_net_state_dict'])
    learner.joint_critic_optimizer.load_state_dict(ckpt['joint_critic_optimizer_state_dict'])

    # load random states in all workers
    for idx, parent in enumerate(envs_runner.parents):
        PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_env_rand_states_" + str(idx) + "1.tar"
        rand_states = torch.load(PATH)
        parent.send(('load_rand_states', rand_states))

    # load actor and ciritc models
    for idx, agent in enumerate(controller.agents):
        PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_agent_" + str(idx) + "1.tar"
        ckpt = torch.load(PATH)
        agent.actor_net.load_state_dict(ckpt['actor_net_state_dict'])
        agent.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])

    return epi_count

if __name__ == '__main__':
    ex.run_commandline()
