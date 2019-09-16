# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch

from .noise import noise


class NormalProbRatio(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, sigma, descriptors, decode_fn):
        ctx.save_for_backward(mu)
        ctx.sigma = sigma
        ctx.descriptors = descriptors
        ctx.decode_fn = decode_fn
        res = torch.ones(len(descriptors), dtype=torch.float32)
        res.requires_grad = True
        return res

    @staticmethod
    def backward(ctx, grad_output):
        mu, = ctx.saved_tensors
        theta = ctx.decode_fn(ctx.descriptors)
        grad = (theta - mu) / ctx.sigma ** 2 * grad_output.unsqueeze(1)
        return (grad, None, None, None)


class MixNormalProbRatio(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mus, sigma, descriptors, decode_fn):
        ctx.save_for_backward(mus)
        ctx.sigma = sigma
        ctx.descriptors = descriptors
        ctx.decode_fn = decode_fn
        res = torch.ones(len(descriptors), dtype=torch.float32)
        res.requires_grad = True
        return res

    @staticmethod
    def backward(ctx, grad_output):
        mus, = ctx.saved_tensors
        thetas = ctx.decode_fn(ctx.descriptors)
        epsilons = [thetas - mu for mu in mus]
        grads = torch.stack(
            [
                (epsilon / ctx.sigma ** 2)
                / (
                    1
                    + sum(
                        torch.exp(
                            -0.5
                            * (other.dot(other) - epsilon.dot(epsilon))
                            / ctx.sigma ** 2
                        )
                        for other in epsilons
                        if other is not epsilon
                    )
                )
                * grad_output
                for epsilon in epsilons
            ]
        )
        return (grads, None, None, None)


class Distribution:
    def __init__(self, device, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState()  # pylint: disable=no-member
        self.random_state = random_state
        self.device = device


class Normal(Distribution):
    def __init__(self, mu, sigma, random_state=None):
        """
        mu: torch.tensor
        sigma: torch.tensor or float
        """
        super().__init__(mu.device, random_state)
        self.mu = mu
        self.sigma = sigma

    def ratio(self, descriptors):
        return NormalProbRatio.apply(self.mu, self.sigma, descriptors, self.decode)

    def sample(self, n, encode=False):
        n_epsilons = n
        noise_inds = np.asarray(
            [
                noise.sample_index(self.random_state, len(self.mu))
                for _ in range(n_epsilons)
            ],
            dtype="int",
        )
        descriptors = [(idx, 1) for idx in noise_inds]
        if encode:
            return descriptors
        thetas = torch.stack([self.decode(descriptor) for descriptor in descriptors])
        return thetas

    def decode(self, descriptor):
        if not isinstance(descriptor, tuple):
            # assert isinstance(descriptor, torch.tensor)
            return descriptor
        noise_idx, direction = descriptor
        epsilon = torch.tensor(noise.get(noise_idx, len(self.mu)), device=self.device)
        with torch.no_grad():
            return self.mu + direction * self.sigma * epsilon


class PairedNormal(Normal):
    def sample(self, n, encode=False):
        assert n % 2 == 0
        n_epsilons = n // 2
        noise_inds = np.asarray(
            [
                noise.sample_index(self.random_state, len(self.mu))
                for _ in range(n_epsilons)
            ],
            dtype="int",
        )
        descriptors = [(idx, 1) for idx in noise_inds] + [
            (idx, -1) for idx in noise_inds
        ]
        if encode:
            return descriptors
        thetas = torch.stack([self.decode(descriptor) for descriptor in descriptors])
        return thetas


class MixNormal(Distribution):
    def __init__(self, mus, sigma, random_state=None):
        """
        mu: torch.tensor
        sigma: torch.tensor or float
        """
        super().__init__(mus[0].device, random_state)
        self.mus = [mu for mu in mus]
        self.sigma = sigma

    def ratio(self, descriptor):
        return MixNormalProbRatio.apply(self.mus, self.sigma, descriptor, self.decode)

    def sample(self, n, encode=False):
        n_epsilons = n
        noise_ids = np.asarray(
            [
                noise.sample_index(self.random_state, len(self.mus[0]))
                for _ in range(n_epsilons)
            ],
            dtype="int",
        )
        mu_ids = np.random.randint(len(self.mus), size=n)
        descriptors = [
            (noise_id, mu_id, 1) for noise_id, mu_id in zip(noise_ids, mu_ids)
        ]
        if encode:
            return descriptors
        thetas = torch.stack([self.decode(descriptor) for descriptor in descriptors])
        return thetas

    def decode(self, descriptor):
        if not isinstance(descriptor, tuple):
            # assert isinstance(descriptor, torch.tensor)
            return descriptor
        noise_id, mu_id, direction = descriptor
        epsilon = torch.tensor(
            noise.get(noise_id, len(self.mus[0])), device=self.device
        )
        with torch.no_grad():
            return self.mus[mu_id] + direction * self.sigma * epsilon
