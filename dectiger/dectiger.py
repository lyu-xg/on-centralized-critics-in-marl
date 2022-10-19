import pandas as pd
import operator
import functools
from collections import Counter
import itertools as itt


states = ['L', 'R']
actions = ['L', 'R']
observations = ['L', 'R']

histories = list(itt.product(observations, repeat=2))
joint_actions = list(itt.product(actions, repeat=2))
joint_histories = list(itt.product(histories, repeat=2))


# Compute Pr(joint_history | s) dataframe


def history_probability(state, history) -> float:
    """computes the probability of an agent history given the state"""
    observation_counts = Counter(L=0, R=0)
    observation_counts.update(history)

    if state == 'L':
        return 0.85 ** observation_counts['L'] * 0.15 ** observation_counts['R']

    if state == 'R':
        return 0.85 ** observation_counts['R'] * 0.15 ** observation_counts['L']

    raise ValueError(f'invalid state {state}')


def joint_history_probability(state, joint_history) -> float:
    """computes the probability of agent histories given the state"""
    probabilities = [history_probability(state, history) for history in joint_history]
    return functools.reduce(operator.mul, probabilities)


data = [(s, joint_history, joint_history_probability(s, joint_history)) for s, joint_history in itt.product(states, joint_histories)]
joint_history_given_state_df = pd.DataFrame(data, columns=['state', 'joint_history', 'joint_history_probability'])
print()
print('Pr(joint history given state)')
# print(joint_history_given_state_df)
print(joint_history_given_state_df.pivot(index='joint_history', columns='state', values='joint_history_probability'))


# Compute Pr(joint_action | joint_history) dataframe


def policy_probability(history, action) -> float:
    """computes the probability of an agent opening a door given its history"""
    observation_counts = Counter(L=0, R=0)
    observation_counts.update(history)

    if action == 'L':
        return observation_counts['R'] / len(history)

    if action == 'R':
        return observation_counts['L'] / len(history)

    raise ValueError(f'invalid action {action}')


def joint_policy_probability(joint_history, joint_action) -> float:
    """computes the joint probability of agents opening doors given their history"""
    probabilities = [policy_probability(history, action) for history, action in zip(joint_history, joint_action)]
    return functools.reduce(operator.mul, probabilities, 1.0)


data = [(joint_history, joint_action, joint_policy_probability(joint_history, joint_action)) for joint_history, joint_action in itt.product(joint_histories, joint_actions)]
joint_action_given_joing_history_df = pd.DataFrame(data, columns=['joint_history', 'joint_action', 'joint_action_probability'])
print()
print('Pr(joint action given joint history)')
# print(joint_action_given_joing_history_df)
print(joint_action_given_joing_history_df.pivot(index='joint_history', columns='joint_action', values='joint_action_probability'))


# Compute Pr(joint_action | state) dataframe


merged_df = pd.merge(joint_history_given_state_df, joint_action_given_joing_history_df, on='joint_history')
merged_df['product_probability'] = merged_df['joint_history_probability'] * merged_df['joint_action_probability']
joint_action_given_state_df = merged_df.groupby(['joint_action', 'state'])['product_probability'].sum().to_frame().reset_index()
print()
print('Pr(joint action given state)')
# print(joint_action_given_state_df)
print(joint_action_given_state_df.pivot(index='joint_action', columns='state', values='product_probability'))
