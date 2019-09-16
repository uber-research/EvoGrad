# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import numpy as np

from evograd import expectation
from evograd.distributions import Normal

import gym


def simulate_single(weights):
    total_reward = 0.0
    num_run = 10
    for t in range(num_run):
        observation = env.reset()
        for i in range(300):
            action = 1 if np.dot(weights, observation) > 0 else 0
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
    return total_reward / num_run


def simulate(batch_weights):
    rewards = []
    for weights in batch_weights:
        rewards.append(simulate_single(weights.numpy()))
    return torch.tensor(rewards)


mu = torch.randn(4, requires_grad=True)  # population mean
npop = 50  # population size
std = 0.5  # noise standard deviation
alpha = 0.03  # learning rate
p = Normal(mu, std)
env = gym.make("CartPole-v0")

for t in range(2000):
    sample = p.sample(npop)
    fitnesses = simulate(sample)
    scaled_fitnesses = (fitnesses - fitnesses.mean()) / fitnesses.std()
    mean = expectation(scaled_fitnesses, sample, p=p)
    mean.backward()

    with torch.no_grad():
        mu += alpha * mu.grad
        mu.grad.zero_()

    print("step: {}, mean fitness: {:0.5}".format(t, float(fitnesses.mean())))
