#!/usr/bin/env python
import torch

from envs import new_env
from configs import ex
from utils import Runner, Team, JointTrajectoryBuffer
from utils.evaluator import Evaluator


@ex.automain
def main(seed, total_rollouts):
    torch.manual_seed(seed)

    env = new_env()
    exp_buffer = JointTrajectoryBuffer(env)

    team = Team(exp_buffer, env)
    runner = Runner(exp_buffer, team, env)
    evaluator = Evaluator(team)

    for _ in range(total_rollouts):
        runner.rollout()
        team.learn()
        evaluator.evaluate()
