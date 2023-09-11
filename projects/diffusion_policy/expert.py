from stanza.runtime import activity
from stanza.rl import ACPolicy
from stanza.rl.nets import MLPActorCritic
from stanza.util.rich import StatisticsTable, ConsoleDisplay, LoopProgress
from stanza import Partial
from stanza.util.logging import logger
from stanza.rl.ppo import PPO
from stanza.train import Trainer
from stanza.data.trajectory import Timestep
from stanza.data import Data
import optax

import stanza.envs as envs
import stanza.policies as policies
import jax

from jax.random import PRNGKey
from stanza.dataclasses import dataclass, replace

@dataclass
class ExpertConfig:
    env_name: str = "brax/ant"
    expert_name: str = None
    episode_length: int = 1000
    total_timesteps: int = 10_000_000

@activity(ExpertConfig)
def train(config, db):
    if config.expert_name is None:
        config = replace(config, expert_name=config.env_name.replace("/","_"))
    exp = db.open("experts").open(config.expert_name)
    logger.info(f"Logging results to {exp.name}")
    env = envs.create(config.env_name)

    net = MLPActorCritic(env.sample_action(PRNGKey(0)))
    params = net.init(PRNGKey(42),
        env.observe(env.sample_state(PRNGKey(0))))

    display = ConsoleDisplay()
    display.add("ppo", StatisticsTable(), interval=1)
    display.add("ppo", LoopProgress("RL"), interval=1)

    ppo = PPO(
        episode_length=config.episode_length,
        total_timesteps=config.total_timesteps,
        trainer = Trainer(
            optimizer=optax.chain(
                optax.clip_by_global_norm(0.5),
                optax.adam(3e-4, eps=1e-5)
            )
        )
    )
    logger.info("Training PPO expert...")
    with display as dh:
        trained_params = ppo.train(
            rng_key=PRNGKey(42),
            env=env,
            ac_apply=net.apply,
            init_params=params,
            rl_hooks=[dh.ppo]
        )
    logger.info("Done training PPO expert.")
    exp.add("config", config)
    exp.add("net", net)
    exp.add("fn_params", trained_params.fn_params)
    exp.add("fn_state", trained_params.fn_state)
    exp.add("obs_normalizer", trained_params.obs_normalizer)

@dataclass
class GenerateConfig:
    expert_name: str = None
    data_name: str = None
    traj_length: int = 1000
    trajectories: int = 1000
    rng_seed: int = 42
    include_jacobian: bool = True

@activity(GenerateConfig)
def generate_data(config, db):
    if config.expert_name is None:
        logger.error("Must specify expert name")
        return
    exp = db.open(f"experts/{config.expert_name}")
    if not "config" in exp:
        print(exp.children)
        logger.error("Expert not found")
        return
    expert_config = exp.get("config")
    expert_net = exp.get("net")
    expert_params = exp.get("fn_params")
    expert_state = exp.get("fn_state")
    normalizer = exp.get("obs_normalizer")

    if config.data_name is None:
        config = replace(config, data_name=config.expert_name)
    data = db.open(f"expert_data/{config.data_name}")
    logger.info(f"Saving to {data.name} using {exp.name}")

    ac_apply = expert_net.apply
    ac_apply = Partial(ac_apply, expert_params)
    env = envs.create(expert_config.env_name)
    policy = ACPolicy(ac_apply, normalizer)

    def rollout(rng_key):
        x0_rng, policy_rng, env_rng = jax.random.split(rng_key, 3)
        roll = policies.rollout(env.step, 
            env.reset(x0_rng), 
            policy, length=config.traj_length,
            policy_rng_key=policy_rng,
            model_rng_key=env_rng,
            observe=env.observe)
        return Data.from_pytree(Timestep(
            roll.observations, roll.actions, 
            roll.states, roll.info))
    trajectories = jax.vmap(rollout)(jax.random.split(PRNGKey(config.rng_seed), config.trajectories))
    trajectories = Data.from_pytree(trajectories)
    data.add("env_name", expert_config.env_name)
    data.add("trajectories", trajectories)