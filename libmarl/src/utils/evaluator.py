import wandb
from envs import new_env
from configs import (
    ex,
    get_number_of_eval_episodes,
    get_trace_len,
    get_discount_factor,
    get_eval_discount,
)
from utils.team import Team
from collections import deque
from itertools import count
from statistics import mean


class Evaluator:
    def __init__(self, team: Team):
        self.env = new_env()
        self.team = team
        self.running_reward = None

    def evaluate(self):
        R, L = [], []
        for _ in range(get_number_of_eval_episodes()):
            h = deque()
            h.append(self.env.reset())
            total_reward = 0.0
            discount = 1.0
            for i_step in count():
                actions = True
                obs, reward, is_done = self.env.step(self.team.act(list(h)))
                total_reward += reward * discount
                if is_done:
                    break
                if get_eval_discount():
                    discount = get_discount_factor() * discount
                h.append(obs)
                if len(h) > get_trace_len():
                    h.popleft()
            R.append(total_reward)
            L.append(i_step)

        r, l = mean(R), mean(L) + 1
        ex.log_scalar("eval_reward", r)
        ex.log_scalar("eval_ep_len", l)
        wandb.log({"eval_reward": r, "eval_ep_len": l})

        discount_msg = "_discounted" if get_eval_discount() else ""
        print(f"Eval_R{discount_msg} {r:.2f}\tEval_ep_len {l:.2f}")
