from stanza.runtime import activity
from stanza.runtime.database import Video
from stanza import Partial
import stanza

from stanza.train import Trainer
from stanza.train.ema import EmaHook
from stanza.train.rich import RichReporter
from stanza.train.wandb import WandbReporter

from stanza.util.dataclasses import dataclass, field
from stanza.util.random import PRNGSequence
from stanza.util.logging import logger
from stanza.model.unet1d import ConditionalUnet1D
from stanza.model.diffusion import DDPMSchedule

from stanza.data.trajectory import chunk_trajectory
from stanza.data.normalizer import LinearNormalizer
from stanza.data import PyTreeData

from rich.progress import track

from functools import partial
from jax.random import PRNGKey

import stanza.policies as policies
import stanza.envs as envs

import jax.numpy as jnp
import jax.random

import stanza.util
import haiku as hk

import optax
import time
import wandb

@dataclass(frozen=True)
class Config:
    env: str = "pusht"
    wandb: str = None # "diffusion_policy"
    rng_seed: int = 42
    epochs: int = 300
    batch_size: int = 256
    warmup_steps: int = 500
    smoothing_sigma: float = 0

def make_network(config):
    if config.env == "pusht":
        from diffusion_policy.networks import pusht_net
        return pusht_net

def make_policy_transform(config, chunk_size=8):
    if config.env == "pusht":
        import stanza.envs.pusht as pusht
        return policies.chain_transforms(
            policies.ChunkTransform(input_chunk_size=2,
                        output_chunk_size=chunk_size),
            # Low-level position controller runs 20x higher
            # frequency than the high-level controller
            # which outputs target positions
            policies.SampleRateTransform(control_interval=10),
            # The low-level controller takes a
            # target position and runs feedback gains
            pusht.PositionObsTransform(),
            pusht.PositionControlTransform()
        )
    raise RuntimeError("Unknown env")

def make_diffuser(config):
    return DDPMSchedule.make_squaredcos_cap_v2(
        100, clip_sample_range=1)

def setup_data(config):
    if config.env == "pusht":
        from stanza.envs.pusht import expert_data
        data = expert_data()
        # Load the data into a PyTree
        logger.info("Calculating data normalizers")
        data_flat = PyTreeData.from_data(data.flatten(), chunk_size=4096)
        action_norm = LinearNormalizer.from_data(
            data_flat.map(lambda x: x.action)
        )
        obs_norm = LinearNormalizer.from_data(
            data_flat.map(lambda x: x.observation)
        )

        # chunk the data and flatten
        logger.info("Chunking trajectories")
        data = data.map(partial(chunk_trajectory,
            obs_chunk_size=2, action_chunk_size=16,
            # full overlap with both observations,
            obs_action_overlap=2))
        # Load the data into a PyTree
        data = PyTreeData.from_data(data.flatten(), chunk_size=4096)
        logger.info("Data Loaded!")
        return data, obs_norm, action_norm

def loss(config, net, diffuser, obs_norm, action_norm,
            # these are passed in per training loop:
            params, rng, sample):
    logger.trace("Tracing loss function", only_tracing=True)
    t_sk, n_sk, s_sk = jax.random.split(rng, 3)
    timestep = jax.random.randint(t_sk, (), 0, diffuser.num_steps)
    # We do training in the normalized space!
    action = action_norm.normalize(sample.action)
    obs = obs_norm.normalize(sample.observation)
    if config.smoothing_sigma > 0:
        obs_flat, obs_uf = jax.flatten_util.ravel_pytree(obs)
        obs_flat = obs_flat + config.smoothing_sigma*jax.random.normal(s_sk, obs_flat.shape)
        obs = obs_uf(obs_flat)

    noisy_action, noise = diffuser.add_noise(n_sk, action, timestep)
    pred_noise = net.apply(params, rng, noisy_action, timestep, obs)

    pred_flat, _ = jax.flatten_util.ravel_pytree(pred_noise)
    noise_flat, _ = jax.flatten_util.ravel_pytree(noise)

    loss = jnp.mean(jnp.square(pred_flat - noise_flat))
    stats = {
        "loss": loss
    }
    return loss, stats

@activity(Config)
def train_policy(config, database):
    rng = PRNGSequence(config.rng_seed)
    exp = database.open("diffusion_policy").create()
    logger.info(f"Running diffusion policy trainer [blue]{exp.name}[/blue]")
    # policy_builder
    # takes params
    # and returns a full policy
    # that can be evaluated
    data, obs_norm, action_norm = setup_data(config)
    net = make_network(config)

    logger.info("Dataset Size: {} chunks", data.length)
    train_steps = Trainer.total_iterations(data, config.batch_size, config.epochs)
    logger.info("Training for {} steps", train_steps)
    warmup_steps = config.warmup_steps

    lr_schedule = optax.join_schedules(
        [optax.linear_schedule(0, 1e-4, warmup_steps),
         optax.cosine_decay_schedule(1e-4, 
                    train_steps - warmup_steps)],
        [warmup_steps]
    )
    optimizer = optax.adamw(lr_schedule, weight_decay=1e-6)
    trainer = Trainer(
        optimizer=optimizer,
        batch_size=config.batch_size,
        epochs=config.epochs
    )
    # Initialize the network parameters
    sample = data.get(data.start)
    logger.info("Instantiating network...")
    # tab = hk.experimental.tabulate(net,
    #         columns=('module', 'owned_params', 
    #                     'input', 'output', 'params_size', 
    #                     'params_bytes'))(sample.action, jnp.array(1),
    #                                      sample.observation)
    # print(tab)
    t = time.time()
    jit_init = jax.jit(net.init)
    init_params = jit_init(next(rng), sample.action,
                           jnp.array(1), sample.observation)
    logger.info(f"Initialization took {time.time() - t}")
    params_flat, _ = jax.flatten_util.ravel_pytree(init_params)
    logger.info("params: {}", params_flat.shape[0])
    logger.info("Making diffusion schedule")

    diffuser = make_diffuser(config)
    loss_fn = Partial(stanza.partial(loss, config, net),
                    diffuser, obs_norm, action_norm)
    loss_fn = jax.jit(loss_fn)

    ema_hook = EmaHook(
        decay=0.75
    )
    logger.info("Initialized, starting training...")

    wr = WandbReporter()
    rr = RichReporter(iter_interval=50, average_window=100)
    if config.wandb is not None:
        wandb.init(project="diffusion_policy")

    with wr as wcb: 
        with rr as rcb:
            hooks = [ema_hook, rcb]
            if config.wandb is not None:
                hooks.append(wcb)
            results = trainer.train(
                        Partial(loss_fn), 
                        data, next(rng), init_params,
                        hooks=hooks
                    )
    params = results.fn_params
    # get the moving average params
    ema_params = results.hook_states[0]
    # save the final checkpoint
    exp.add('config', config)
    exp.add('obs_norm', obs_norm)
    exp.add('action_norm', action_norm)
    exp.add('final_checkpoint', params)
    exp.add('final_checkpoint_ema', ema_params)

@activity(Config)
def sweep_train(config, database):
    pass

@dataclass
class EvalConfig:
    path: str = None
    rng_key: PRNGKey = PRNGKey(42)
    samples: int = 10
    rng_seed: int = 42


def rollout(env, policy, length, rng):
    x0_rng, policy_rng = jax.random.split(rng)
    x0 = env.reset(x0_rng)
    r = policies.rollout(env.step, x0, policy,
                    policy_rng_key=policy_rng,
                    length=length, last_state=True)
    return r

@activity(EvalConfig)
def eval(eval_config, database, results=None):
    if results is None:
        results = database.open(eval_config.path)
    logger.info(f"Evaluating [blue]{results.name}[/blue]")
    config = results.get("config")
    obs_norm = results.get("obs_norm")
    action_norm = results.get("action_norm")
    params = results.get("final_checkpoint_ema")
    logger.info("Loaded final checkpoint")

    net = make_network(config)
    diffuser = make_diffuser(config)

    logger.info("Creating environment")
    env = envs.create(config.env)
    action_sample = env.sample_action(PRNGKey(0))

    net_fn = Partial(net.apply, params)
    from stanza.policies.diffusion import make_diffusion_policy

    policy = make_diffusion_policy(
        net_fn, diffuser,
        obs_norm, action_norm, action_sample, 
        16, 1, 8
    )

    # import stanza.envs.pusht as pusht
    # obs = pusht.PushTPositionObs(
    #     jnp.array([[200, 200], [202, 202]]),
    #     jnp.array([[100, 100], [100, 100]]),
    #     jnp.array([0, 0])
    # )
    # logger.info("output {}", 
    #     policy(policies.PolicyInput(obs, None, PRNGKey(42))).action)

    policy = make_policy_transform(config)(policy)
    logger.info("Rolling out policies...")
    rollout_fn = partial(rollout, env, policy, 300*10)
    mapped_rollout_fun = jax.jit(jax.vmap(rollout_fn))
    rngs = jax.random.split(PRNGKey(eval_config.rng_seed), eval_config.samples)
    r = mapped_rollout_fun(rngs)
    logger.info("Computing scores...")
    eval_states = jax.tree_util.tree_map(lambda x: x[:,::10], r.states)
    scores = jax.vmap(jax.vmap(env.score))(eval_states)
    # get the highest coverage over the sample
    scores = jnp.max(scores,axis=1)
    logger.info("Scores: {}", scores)
    logger.info("Mean scores: {}", scores.mean())

    logger.info("Persisting eval data...")
    vis_states = jax.tree_util.tree_map(lambda x: x[:, ::10], r.states)
    video = jax.vmap(jax.vmap(env.render))(vis_states)
    scores = results.open("samples")
    logger.info("Writing videos...")
    for i in range(video.shape[0]):
        scores.add(f"sample_{i}", Video(video[i],fps=28))
    logger.info("Done!")