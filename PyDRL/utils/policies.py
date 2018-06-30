import numpy as np
import random

def RandomPolicy(num_actions):
	return random.randrange(num_actions)

def GreedyPolicy(num_actions, q_values):
	return np.argmax(q_values[0])

def GreedyEpsilonPolicy(num_actions, q_values, epsilon):
	if np.random.rand() <= epsilon:
            return random.randrange(num_actions)
    return np.argmax(q_values[0])