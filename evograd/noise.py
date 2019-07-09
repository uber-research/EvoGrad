#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.


import logging

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

debug = True


class SharedNoiseTable:
    def __init__(self):
        import ctypes
        import multiprocessing

        seed = 42
        # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        count = 250000000 if not debug else 10000000
        logger.info("Sampling {} random numbers with seed {}".format(count, seed))
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(  # pylint: disable=no-member
            count
        )  # 64-bit to 32-bit conversion here
        logger.info("Sampled {} bytes".format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i : i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)


noise = SharedNoiseTable()
