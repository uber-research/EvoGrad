#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.


import torch

from evograd import expectation
from evograd.distributions import Normal


def fun(x):
    return 5 * torch.sin(0.2 * x) * torch.sin(20 * x)


mu = torch.tensor([1.0], requires_grad=True)
npop = 500  # population size
std = 0.5  # noise standard deviation
alpha = 0.03  # learning rate
p = Normal(mu, std)

for t in range(2000):
    sample = p.sample(npop)
    fitnesses = fun(sample)
    fitnesses = (fitnesses - fitnesses.mean()) / fitnesses.std()
    mean = expectation(fitnesses, sample, p=p)
    mean.backward()

    with torch.no_grad():
        mu += alpha * mu.grad
        mu.grad.zero_()

    print('step: {}, mean fitness: {:0.5}'.format(t, float(mu)))
