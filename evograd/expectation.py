# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.


import torch


def unsqueeze_as(x, y):
    assert len(y.shape) >= len(x.shape)
    for _ in range(len(y.shape) - len(x.shape)):
        x = torch.unsqueeze(x, -1)
    return x


def expectation(vals, sample, p):
    ratio = unsqueeze_as(p.ratio(sample), vals)
    prod = vals * ratio
    return prod.mean(dim=0)
