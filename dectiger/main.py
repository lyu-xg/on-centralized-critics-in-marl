import numpy as np
from collections import Counter, defaultdict
from itertools import product, chain
from rl_parsers.dpomdp import parse


class DecTiger():
	def __init__(self, horizon=2):
		with open('dectiger.dpomdp') as f:
			self.d = parse(f.read())

		self.horizon = horizon
		self.n_agents = len(self.d.agents)
		self.states = self.d.states
		self.actions = self.d.actions[0]
		self.observations = self.d.observations[0]
		self.joint_observations = tuple(
			product(self.observations, repeat=self.n_agents))
		self.joint_actions = tuple(product(self.actions, repeat=self.n_agents))

		# (state, iteration, history_objects)
		self.histories = [[] for _ in self.states]
		# history object: ([action-observations], Pr(h|s), cul_reward)
		for s in self.states:
			self.grow_history(s)

		self.state_history_values = [[] for _ in self.states]
		for s in range(len(self.states)):
			self.gen_state_history_values(s)

		self.prob_h = self.get_history_probs()
		self.prob_s_given_h = self.get_state_given_hist_probs()

		self.state_values = self.gen_state_values()
		self.history_values = self.gen_history_values()

	def transition(self, state, actions):
		a = self.actions2a(actions)
		s = self.states.index(state)
		obs_probs = [self.d.O[(*a, s, *self.obs2o(obs))]
                    for obs in self.joint_observations]
		reward = [float(self.d.R[(*a, s, s, *self.obs2o(obs))])
                    for obs in self.joint_observations]
		return obs_probs, reward

	def actions2a(self, actions):
		return [self.actions.index(action) for action in actions]

	def obs2o(self, observations):
		return [self.observations.index(obs) for obs in observations]

	def grow_history(self, state, histories=None, horizon=None):
		# dectiger states are self-transitioning, so we do not consider state transitions
		# history will be associated with a probability in a tuple,
		# and branch out according to the transition table
		# default to empty history set with the only empty history with probability 1 and zero reward
		if histories is None:
			histories = [([], 1, 0)]
		if horizon is None:
			horizon = self.horizon
		if horizon <= 0:
			return histories
		frontier = []
		for history, hist_prob, hist_reward in histories:
			a = self.policy(history)
			observations_probs, rewards = self.transition(state, a)
			for p, o, r in zip(observations_probs, self.joint_observations, rewards):
				if p == 0:
					continue
				# the dectiger reward model is determinstic given the joint actions,
				# so no branching happens at assigning rewards
				frontier.append((history + [a, o], hist_prob * p, hist_reward+r))
		self.grow_history(state, frontier, horizon-1)
		frontier = [(h, p/self.horizon, r) for h, p, r in frontier]
		self.histories[self.states.index(state)].append(frontier)

	def get_history_probs(self):
		res = defaultdict(list)
		for s, values in zip(self.states, self.state_history_values):
			for h, p, q in chain(*values):
				res[h].append(p)

		def hist_val(P):
			return sum(p * 1/2 for p in P)  # P(h) = \sum_s P(h|s) P(s) where P(s) = 1/2
		return {h: hist_val(P) for h, P in res.items()}

	def get_state_given_hist_probs(self):
		# returns dictionary of P(s|h)
		res = dict()
		for s, values in zip(self.states, self.state_history_values):
			for h, p, q in chain(*values):
				# Pr(s|h) = (Pr(h|s)Pr(s)) / Pr(h)
				res[(s, h)] = (p * 1/2) / self.prob_h[h]
		return res

	def policy(self, history):
		if len(history)/2 < self.horizon - 1:
			return ('listen', 'listen')

		def local_a(O):
			return 'open-right' if Counter(O).most_common(1)[0][0] == 'hear-left' else 'open-left'
		return tuple(local_a(o) for o in zip(*history[1::2]))

	def gen_state_history_values(self, s):
		# bottom up approach to calculate all the history values
		# collected probabilities and rewards
		# state_history_values: (state, depth, state-history-value-object)
		# state-history-value-object: ([action-observations], p(h|s), Q(s,h,a))
		def get_prob_and_expected_value(prob_and_value):
			p, r = zip(*prob_and_value)
			normal_p = np.array(p) / sum(p)
			return sum(p), sum(normal_p * np.array(r))
		frontiers = self.histories[s]
		values = frontiers[0]
		# self.history_values[s].append(values)
		last_obs = True
		while len(values) > 1:
			collected = defaultdict(list)
			for h, p, r in values:
				prev_h = tuple(h[:-1]) if last_obs else tuple(h[:-2])
				collected[prev_h].append((p, r))
			last_obs = False
			values = [(h, *get_prob_and_expected_value(p_v))
                            for h, p_v in collected.items()]
			self.state_history_values[s].append(values)

	def gen_state_values(self):
		def Qs(hs, a):
			filtered_state_history_q_values = list(
				filter(lambda h: a in h[0], chain(*hs)))
			if not filtered_state_history_q_values:
				return None
			_, p, r = zip(*filtered_state_history_q_values)
			p, r = np.array(p), np.array(r)
			p = p/sum(p)  # normalize
			return sum(p * r)
		# need to use history values
		return {(s, a): Qs(hs, a) for s, hs in zip(self.states, self.state_history_values) for a in self.joint_actions}

	def gen_history_values(self):
		res = dict()
		collected = defaultdict(list)
		for s, values in zip(self.states, self.state_history_values):
			for h, _, q in chain(*values):
				# h,a = ha[:-1], ha[-1]
				collected[h].append((s, q))
		for h, s_q in collected.items():
			S, q = zip(*s_q)
			p = [self.prob_s_given_h[(s, h)] for s in S]
			res[h] = sum(np.array(p) * np.array(q))
		return res

	def get_approximated_history_values(self):
		# returns \sum_s P(s|h) Q(s, a)
		res = dict()
		collected = defaultdict(list)
		for s, values in zip(self.states, self.state_history_values):
			for ha, _, _ in chain(*values):
				h, a = ha[:-1], ha[-1]
				q = self.state_values[(s, a)]
				p = self.prob_s_given_h[(s, ha)]
				collected[ha].append((p, q))
		for h, p_q in collected.items():
			p, q = zip(*p_q)
			res[h] = sum(np.array(p) * np.array(q))
		return res

	def get_h_gradient(self, h_i):
		H = {h: p for h, p in self.prob_h.items() if self.contain(h, h_i)}
		normalize_constraint = sum(H.values())
		for h, p in H.items():
			print(h, p/normalize_constraint, self.history_values[h])
		return sum((p/normalize_constraint)*self.history_values[h] for h, p in H.items())

	@staticmethod
	def contain(h, h_i):
		return len(h) == len(h_i) and all(h_t[0] == h_i_t for (h_t, h_i_t) in zip(h, h_i))

	def get_s_gradient(self, h_i):
		approximated_history_values = self.get_approximated_history_values()
		H = {h: p for h, p in self.prob_h.items() if self.contain(h, h_i)}
		normalize_constraint = sum(H.values())
		for h, p in H.items():
			print(h, p/normalize_constraint, approximated_history_values[h])
		return sum((p/normalize_constraint)*approximated_history_values[h] for h, p in H.items())

	def list_h_i(self):
		return [tuple(h_t[0] for h_t in h) for h in self.prob_h.keys()]
