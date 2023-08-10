import stanza

from stanza.runtime import activity
from stanza.reporting import Video
from stanza import Partial

from stanza.data.trajectory import Timestep
from stanza.data import Data
from stanza.train import Trainer, batch_loss
from stanza.train.ema import EmaHook
from stanza.train.validate import Validator

from stanza.policies.mpc import MPC
from stanza.solver.ilqr import iLQRSolver

from stanza.dataclasses import dataclass, replace, field
from stanza.util.attrdict import AttrMap
from stanza.util.random import PRNGSequence
from stanza.util.loop import LoggerHook, every_kth_epoch, every_kth_iteration
from stanza.util.logging import logger
from stanza.util.rich import ConsoleDisplay, \
    LoopProgress, StatisticsTable, EpochProgress
from stanza.diffusion.ddpm import DDPMSchedule

from stanza.data.trajectory import chunk_trajectory
from stanza.data.normalizer import LinearNormalizer
from stanza.data import PyTreeData

from rich.progress import track

from functools import partial
from jax.random import PRNGKey

import stanza.policies as policies
import stanza.policies.transforms as transforms
import stanza.envs as envs

import jax.numpy as jnp
import jax.random

import stanza.util

import optax
import time

@dataclass
class Config:
    use_gains: bool = False
    env: str = "pusht"
    wandb: str = "diffusion_policy"
    rng_seed: int = 42
    epochs: int = 500
    batch_size: int = 256
    warmup_steps: int = 500
    num_datapoints: int = 100
    smoothing_sigma: float = 0

def make_network(config):
    if config.env == "pusht":
        from diffusion_policy.networks import pusht_net
        return pusht_net
    elif config.env == "quadrotor":
        from diffusion_policy.networks import quadrotor_net
        return quadrotor_net

def make_policy_transform(config, chunk_size=8):
    if config.env == "pusht":
        import stanza.envs.pusht as pusht
        return transforms.chain_transforms(
            transforms.ChunkTransform(input_chunk_size=2,
                        output_chunk_size=chunk_size),
            # Low-level position controller runs 20x higher
            # frequency than the high-level controller
            # which outputs target positions
            transforms.SampleRateTransform(control_interval=10),
            # The low-level controller takes a
            # target position and runs feedback gains
            pusht.PositionObsTransform(),
            pusht.PositionControlTransform()
        )
    elif config.env == "quadrotor":
        t = [
            transforms.ChunkTransform(
                input_chunk_size=2,
                output_chunk_size=chunk_size
            )
        ]
        if config.use_gains:
            t.append(transforms.FeedbackTransform())
        return transforms.chain_transforms(*t)
    raise RuntimeError("Unknown env")

def make_diffuser(config):
    return DDPMSchedule.make_squaredcos_cap_v2(
        100, clip_sample_range=10.)

def setup_data(config, rng_key):
    if config.env == "pusht":
        from stanza.envs.pusht import expert_data
        data = expert_data()
        traj = min(200,
            config.num_datapoints \
                if config.num_datapoints is not None else 200)
        val_data = data[200:]
        data = data[:traj]
        # Load the data into a PyTree
    elif config.env == "quadrotor":
        env = envs.create("quadrotor")
        # rollout a bunch of trajectories
        mpc = MPC(
            action_sample=env.sample_action(PRNGKey(0)),
            cost_fn=env.cost,
            model_fn=env.step,
            horizon_length=100,
            receed=False,
            solver=iLQRSolver()
        )
        from stanza.solver.ilqr import linearize, tvlqr
        def rollout(rng_key):
            rng = PRNGSequence(rng_key)
            x0 = env.reset(next(rng))
            rollout = policies.rollout(env.step, x0, mpc,
                            length=100, last_state=False)
            As, Bs = jax.vmap(linearize(env.step))(rollout.states, rollout.actions, None)
            Ks = tvlqr(As, Bs, jnp.eye(As.shape[-1]), jnp.eye(Bs.shape[-1]))
            Ks = jnp.zeros_like(Ks)
            return Data.from_pytree(Timestep(
                rollout.states, rollout.actions, rollout.states,
                info=AttrMap(K=Ks)
            ))
        data = jax.vmap(rollout)(jax.random.split(rng_key, config.num_datapoints + 20))
        data = Data.from_pytree(data)
        val_data = data[-20:]
        data = data[:-20]
    logger.info("Calculating data normalizers")
    data_flat = PyTreeData.from_data(data.flatten(), chunk_size=4096)

    normalizer = LinearNormalizer.from_data(data_flat)
    # chunk the data and flatten
    logger.info("Chunking trajectories")
    def slice_chunk(x):
        return replace(x,
            observation=jax.tree_util.tree_map(
                lambda x: x[:2],
                x.observation
            )
        )
    def chunk(traj):
        traj = chunk_trajectory(traj, 16, 1, 7)
        return traj.map(slice_chunk)
    data = data.map(chunk)
    val_data = val_data.map(chunk)

    # Load the data into a PyTree
    data = PyTreeData.from_data(data.flatten(), chunk_size=4096)
    val_data = PyTreeData.from_data(val_data.flatten(), chunk_size=4096)
    logger.info("Data Loaded!")
    return data, val_data, normalizer

def loss(config, net, diffuser, normalizer,
            # these are passed in per training loop:
            state, params, rng, sample):
    logger.trace("Tracing loss function", only_tracing=True)
    rng = PRNGSequence(rng)
    timestep = jax.random.randint(next(rng), (), 0, diffuser.num_steps)
    # We do training in the normalized space!
    normalized_sample = normalizer.normalize(sample)

    # the state/action chunks
    actions = normalized_sample.action
    states = normalized_sample.state
    # the observation chunks (truncated)
    obs = normalized_sample.observation

    if config.smoothing_sigma > 0:
        obs_flat, obs_uf = jax.flatten_util.ravel_pytree(obs)
        obs_flat = obs_flat + config.smoothing_sigma*jax.random.normal(next(rng), obs_flat.shape)
        obs = obs_uf(obs_flat)

    if config.use_gains:
        input = actions, states, normalized_sample.info.K
    else:
        input = actions, None, None

    noisy, noise = diffuser.add_noise(next(rng), input, timestep)
    pred_noise = net.apply(params, next(rng), noisy, timestep, obs)
    pred_flat, _ = jax.flatten_util.ravel_pytree(pred_noise)
    noise_flat, _ = jax.flatten_util.ravel_pytree(noise)
    loss = jnp.mean(jnp.square(pred_flat - noise_flat))

    stats = {
        "loss": loss
    }
    return state, loss, stats

@activity(Config)
def train_policy(config, database):
    from stanza.reporting.wandb import WandbDatabase
    db = WandbDatabase("dpfrommer-projects/diffusion_policy")
    db = db.create()

    rng = PRNGSequence(config.rng_seed)
    exp = database.open("diffusion_policy").open(db.name)
    logger.info(f"Running diffusion policy trainer [blue]{exp.name}[/blue]")

    # load the data to the CPU, then move to the GPU
    with jax.default_device(jax.devices("cpu")[0]):
        data, val_data, normalizer = setup_data(config, next(rng))
    # move data to GPU:
    data, val_data, normalizer = \
        jax.device_put((data, val_data, normalizer),
                       jax.devices()[0])
    net = make_network(config)

    logger.info("Dataset Size: {} chunks", data.length)
    train_steps = (data.length // config.batch_size) * config.epochs
    logger.info("Training for {} steps", train_steps)
    warmup_steps = config.warmup_steps

    lr_schedule = optax.join_schedules(
        [optax.linear_schedule(1e-4/500, 1e-4, warmup_steps),
         optax.cosine_decay_schedule(1e-4, 
                    train_steps - warmup_steps)],
        [warmup_steps]
    )
    optimizer = optax.adamw(lr_schedule, weight_decay=1e-5)
    trainer = Trainer(
        optimizer=optimizer,
        batch_size=config.batch_size,
        epochs=config.epochs
    )
    sample = data.get(data.start)
    logger.info("Instantiating network...")
    t = time.time()
    jit_init = jax.jit(net.init)

    if config.use_gains:
        sample_input = sample.action, sample.state, sample.info.K
    else:
        sample_input = sample.action, None, None
    init_params = jit_init(next(rng), sample_input,
                           jnp.array(1), sample.observation)
    logger.info(f"Initialization took {time.time() - t}")
    params_flat, _ = jax.flatten_util.ravel_pytree(init_params)
    logger.info("params: {}", params_flat.shape[0])
    logger.info("Making diffusion schedule")

    diffuser = make_diffuser(config)
    loss_fn = Partial(stanza.partial(loss, config, net),
                    diffuser, normalizer)
    loss_fn = jax.jit(loss_fn)

    ema_hook = EmaHook(
        decay=0.75
    )
    logger.info("Initialized, starting training...")

    from stanza.reporting.jax import JaxDBScope
    dbs = JaxDBScope(db)
    print_hook = LoggerHook(every_kth_iteration(500))

    display = ConsoleDisplay()
    display.add("train", StatisticsTable(), interval=100)
    display.add("train", LoopProgress(), interval=100)
    display.add("train", EpochProgress(), interval=100)

    validator = Validator(
        condition=every_kth_epoch(1),
        rng_key=next(rng),
        dataset=val_data,
        batch_size=config.batch_size)

    with display as rcb, dbs as dbs:
        stat_logger = dbs.statistic_logging_hook(
            log_cond=every_kth_iteration(1), buffer=100)
        hooks = [ema_hook, validator, rcb.train, 
                    stat_logger, print_hook]
        results = trainer.train(
                    batch_loss(loss_fn), 
                    data, next(rng), init_params,
                    hooks=hooks
                )
    params = results.fn_params
    # get the moving average params
    ema_params = results.hook_states[0]
    # save the final checkpoint
    exp.add('config', config)
    exp.add('normalizer', normalizer)
    exp.add('final_checkpoint', params)
    exp.add('final_checkpoint_ema', ema_params)
    # run the evaluation code as well immediately after
    noise_res, no_noise_res  = eval(EvalConfig(), database, results=exp)
    db.run.summary["replica_error"] = noise_res
    db.run.summary["deconv_error"] = no_noise_res
    db.run.summary["replica_error_mean"] = jnp.mean(noise_res)
    db.run.summary["deconv_error_mean"] = jnp.mean(no_noise_res)

@activity(Config)
def sweep_train(config, database):
    pass

@dataclass
class EvalConfig:
    path: str = None
    rng_key: PRNGKey = field(default_factory=lambda:PRNGKey(42))
    samples: int = 25
    rng_seed: int = 42


def rollout(env, policy, length, x0_rng, policy_rng):
    x0 = env.reset(x0_rng)
    r = policies.rollout(env.step, x0, policy,
                    policy_rng_key=policy_rng,
                    length=length, last_state=True)
    return r

def compute_scores(config, env, results, noise_states, no_noise_states):
    if config.env == "quadrotor":
        mpc = MPC(
            action_sample=env.sample_action(PRNGKey(0)),
            cost_fn=env.cost,
            model_fn=env.step,
            horizon_length=100,
            receed=False,
            solver=iLQRSolver()
        )
        def expert_states(x0):
            rollout = policies.rollout(env.step, x0, mpc,
                            length=100, last_state=True)
            return rollout.states
        x0s = jax.tree_map(lambda x: x[:,0], noise_states)
        expert_states = jax.vmap(expert_states)(x0s)
        from stanza.util import l2_norm_squared

        noise_diff = jax.vmap(jax.vmap(l2_norm_squared))(
            expert_states, noise_states
        )
        noise_diff = jnp.sum(noise_diff, axis=-1)
        no_noise_diff = jax.vmap(jax.vmap(l2_norm_squared))(
            expert_states, no_noise_states
        )
        no_noise_diff = jnp.sum(no_noise_diff, axis=-1)
        logger.info("deconv error: {}", no_noise_diff)
        logger.info("replica error: {}", noise_diff)
        results.add("deconv", no_noise_diff)
        results.add("replica", noise_diff)
        return noise_diff, no_noise_diff
    else:
        logger.info("Computing scores...")
        eval_states = jax.tree_util.tree_map(lambda x: x[:,::10], noise_states)
        scores = jax.vmap(jax.vmap(env.score))(eval_states)
        # get the highest coverage over the sample
        scores = jnp.max(scores,axis=1)
        logger.info("Scores: {}", scores)
        logger.info("Mean scores: {}", scores.mean())

@activity(EvalConfig)
def eval(eval_config, database, results=None):
    if results is None:
        results = database.open(eval_config.path)
    logger.info(f"Evaluating [blue]{results.name}[/blue]")
    config = results.get("config")
    if config is None:
        logger.error("No such result")
        return
    logger.info(f"Loaded config {config}")
    normalizer = results.get("normalizer")
    params = results.get("final_checkpoint_ema")
    logger.info("Loaded final checkpoint")
    params_flat, _ = jax.flatten_util.ravel_pytree(params)
    if jnp.any(jnp.isnan(params_flat)):
        import sys
        logger.error("NaNs in params")
        sys.exit(1)

    net = make_network(config)
    diffuser = make_diffuser(config)

    logger.info("Creating environment")
    env = envs.create(config.env)

    net_fn = Partial(net.apply, params)
    from stanza.policies.diffusion import make_diffusion_policy
    replica_policy = make_diffusion_policy(
        net_fn, diffuser, normalizer, 16, 1, 8,
        diffuse_gains=config.use_gains,
        noise=config.smoothing_sigma
    )
    replica_policy = make_policy_transform(config)(replica_policy)
    deconv_policy = make_diffusion_policy(
        net_fn, diffuser, normalizer, 16, 1, 8,
        diffuse_gains=config.use_gains
    )
    deconv_policy = make_policy_transform(config)(deconv_policy)
    deconv_rollout_fn = partial(rollout, env, deconv_policy, 100)
    replica_rollout_fn = partial(rollout, env, replica_policy, 100)
    logger.info("Rolling out policies...")
    mapped_deconv_fun = jax.jit(jax.vmap(deconv_rollout_fn))
    mapped_replica_fun = jax.jit(jax.vmap(replica_rollout_fn))

    x0_rng, deconv_rng, replica_rng = jax.random.split(eval_config.rng_key, 3)
    x0_rngs = jax.random.split(x0_rng, eval_config.samples)
    deconv_rngs = jax.random.split(deconv_rng, eval_config.samples)
    replica_rngs = jax.random.split(replica_rng, eval_config.samples)
    deconv_rollouts = mapped_deconv_fun(x0_rngs, deconv_rngs)
    replica_rollouts = mapped_replica_fun(x0_rngs, deconv_rngs)
    logger.info("Computing statistics")
    return compute_scores(config, env, results, replica_rollouts.states, deconv_rollouts.states)
