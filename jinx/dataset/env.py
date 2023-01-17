from .rng import RNGDataset

from jinx.policy import SampleRandom

import jinx.envs
import jax

class EnvDataset(RNGDataset):
    def __init__(self, rng_key, env, traj_length, policy=None):
        rng_key, sk = jax.random.split(rng_key)
        super().__init__(rng_key)
        self.env = env
        # If policy is not specified, sample random
        # actions from the environment
        if policy is None:
            policy = SampleRandom(sk, env.sample_action)
        self.policy = policy
        self.traj_length = traj_length

    def get(self, iterator):
        rng = super().get(iterator)

        states, us = jinx.envs.rollout_policy(self.env.step,
            self.env.reset(rng),
            self.traj_length, self.policy)
        return (states, us)