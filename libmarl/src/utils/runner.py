import wandb
import numpy as np
from typing import List
from envs import Env
from configs import (
    ex,
    get_is_state_based_critic,
    get_trace_len,
    get_buffer_size,
    get_method,
    get_episodic_train,
    get_smoothing,
    is_reward_func_aligned,
    get_env,
    get_is_centralized_critic,
    get_is_q_func,
    get_reward_window,
)
from utils.buffer import TrajectoryBuffer
from utils.team import Team
from collections import deque


class Runner:
    def __init__(self, exp_buffer: TrajectoryBuffer, team: Team, env: Env):
        self.exp_buffer, self.team, self.env = exp_buffer, team, env
        self.running_reward = 0.0
        self.ep_reward = 0.0
        self.window_culmulative_reward = 0.0
        self.i_ep, self.i_rollout = 0, 0
        self.cur_ep_len = 0
        self.i_step = 0
        self.reward_window = get_reward_window()
        self.aligned_reward = is_reward_func_aligned()
        self.obs_buffer = deque()

    def rollout(self):
        if self.i_ep == 0:
            self.obs_buffer.append(self.env.reset())
            self.state = self.env.get_state() if get_is_state_based_critic() else None
        self.exp_buffer.reset()
        self.exp_buffer.add_initial_state(self.obs_buffer[-1], self.state)
        trace_len = get_trace_len()

        while len(self.exp_buffer) < get_buffer_size():
            actions = self.team.act(list(self.obs_buffer)[-trace_len:])
            obs, reward, done = self.env.step(actions)
            self.state = self.env.get_state() if get_is_state_based_critic() else None
            self.exp_buffer.append(actions, obs, reward, done, self.state)
            self.ep_reward += reward if self.aligned_reward else np.mean(reward)
            self.step_telemetry(reward, done)
            self.cur_ep_len += 1
            self.obs_buffer.append(obs)
            if done:
                self.ep_end()
                if get_episodic_train():
                    break

        self.end_of_rollout_telemetry()
        ex.log_scalar("running_reward", self.running_reward)
        wandb.log({"running_reward": self.running_reward})
        self.i_rollout += 1

    def ep_end(self):
        # called when reaching the end of an episode
        self.obs_buffer.clear()
        self.obs_buffer.append(self.env.reset())
        self.state = self.env.get_state() if get_is_state_based_critic() else None
        self.exp_buffer.add_initial_state(self.obs_buffer[-1], self.state)

        # if self.i_ep and self.i_ep % 100 == 0:
        #     print(f'Episode {self.i_ep}\tLast reward: {self.ep_reward:.2f}\tRunning reward: {self.running_reward:.2f}')

        smoothing = get_smoothing()
        if self.running_reward is None:
            self.running_reward = (
                self.ep_reward
            )  # first episode as the starting point of the running_reward
        self.running_reward = (
            smoothing * self.ep_reward + (1 - smoothing) * self.running_reward
        )
        # ex.log_scalar('running_reward', self.running_reward)
        # ex.log_scalar('ep_reward', self.ep_reward)
        # ex.log_scalar('ep_len', self.cur_ep_len)
        # wandb.log({'running_reward': self.running_reward, 'ep_reward': self.ep_reward, 'ep_len': self.cur_ep_len})
        self.ep_reward = 0.0
        self.cur_ep_len = 0
        self.i_ep += 1

    def step_telemetry(self, reward, done):
        self.i_step += 1
        if type(reward) is list:
            reward = sum(reward)
        self.window_culmulative_reward += reward
        if self.i_step and self.reward_window and self.i_step % self.reward_window == 0:
            ex.log_scalar("cul_reward", self.window_culmulative_reward)
            wandb.log({"cul_reward": self.window_culmulative_reward})
            self.window_culmulative_reward = 0

    def print_stag_policy(self):
        init_state = np.array([1.0, 0.0, 0.0, 0.0])
        # init_joint_state = np.hstack((init_state, init_state))
        # if get_is_centralized_critic():
        (s1a1, s1a2), (s2a1, s2a2) = self.team.get_action_probs(init_state)
        print(
            f"Episode {self.i_ep}\tRunning reward: {self.running_reward:.2f}\ta0u0: {s1a1:.3f}\ta1u0: {s2a1:.3f}"
        )

    def log_dectiger_value(self):
        states = np.array([[[0]], [[1]]])
        values = self.team.central_critic(states, target_val=False)
        print(values)

    def end_of_rollout_telemetry(self):
        print(
            f"Rollout {self.i_rollout}\tEpisode {self.i_ep}\tLast reward: {self.ep_reward:.2f}\tRunning reward: {self.running_reward:.2f}"
        )

        if get_env() == "stag":
            self.print_stag_policy()
        if get_env() == "dectiger":
            # self.log_dectiger_value()
            pass
        if get_env() != "guess_color":
            return

        joint_states = [[1, 1], [1, 0], [0, 1], [0, 0]]
        if get_is_centralized_critic():
            for (s1, s2), v in zip(
                joint_states, self.team.central_critic(np.array(joint_states))
            ):
                if get_is_q_func():
                    for qi, q in enumerate(v):
                        a1, a2 = self.team.central_critic.joint2actions(qi)
                        ex.log_scalar(f"state_({s1},{s2})_action({a1},{a2})", q.item())
                else:
                    ex.log_scalar(f"state_({s1},{s2})", v.item())
        else:
            for i_agent, values in enumerate(
                self.team.get_values(np.array([[0], [1]]))
            ):
                # for each agent, we need the value of each state here
                for s, v in zip((0, 1), values):
                    if get_is_q_func():
                        for qi, q in enumerate(v):
                            ex.log_scalar(
                                f"agent{i_agent}state{s}action{qi}value", q.item()
                            )
                    else:
                        ex.log_scalar(f"agent{i_agent}state{s}value", v.item())

        for i_agent, action_prob in enumerate(
            self.team.get_action_probs(np.array([[0], [1]]))
        ):
            # for each agent, we need the prob of each action here

            for s, pi in zip((0, 1), action_prob):
                for i_action in (0, 1, 2):
                    ex.log_scalar(
                        f"agent{i_agent}state{s}action{i_action}",
                        action_prob[s][i_action].item(),
                    )
