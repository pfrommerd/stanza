
from functools import partial
from jax.random import PRNGKey
from typing import Any

from stanza.util.dataclasses import dataclass, replace
from stanza.util.logging import logger
from stanza.runtime import activity
from stanza.dataset.env import EnvDataset
from stanza.dataset import PyTreeDataset
from stanza.train import Trainer
from stanza.solver.optax import OptaxSolver
from stanza.solver.newton import NewtonSolver

from stanza.policies.mpc import MPC
from stanza.policies import RandomPolicy

import stanza.envs
import stanza.util.random

import jax
import jax.numpy as jnp
import haiku as hk
import optax

import plotly.express as px


_ACTIVATION_MAP = {
    'gelu': jax.nn.gelu,
    'relu': jax.nn.relu,
    'swish': jax.nn.swish,
    'tanh': jax.nn.tanh,
}

@dataclass
class Config:
    learned_model: bool = True
    lr: float = None
    iterations: int = 25000
    batch_size: int = 100
    hidden_dim: int = None
    hidden_layers: int = 3

    # This is trajectories *worth* of data
    # not actual rollout trajectories
    init_trajectories: int = 100
    total_trajectories: int = 1000
    traj_interval: int = 450

    rng_seed: int = 69
    traj_length: int = 50

    eval_trajs: int = 20

    env: str = "pendulum"

    activation: str = None

    jacobian_regularization: float = 0.0  # 0.0 disables it

    verbose: bool = False
    show_pbar: bool = True

def set_default(config, attr, default):
    if getattr(config, attr) == None:
        setattr(config, attr, default)

def make_solver(gt=False):
    if gt:
        return NewtonSolver()
    optimizer = optax.chain(
        # Set the parameters of Adam. Note the learning_rate is not here.
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        optax.scale_by_schedule(optax.cosine_decay_schedule(1.0,
                                5000, alpha=0.01)),
        # Put a minus sign to *minimise* the loss.
        optax.scale(-0.05)
    )
    return OptaxSolver(optimizer=optimizer, max_iterations=5000)

def map_fn(traj):
    states, actions = traj.states, traj.actions
    prev_states = jax.tree_util.tree_map(lambda x: x[:-1], states)
    next_states = jax.tree_util.tree_map(lambda x: x[1:], states)
    return prev_states, actions, next_states

def generate_dataset(config, env, curr_model_fn, rng_key, num_traj, prev_data):
    if curr_model_fn is None:
        rng_key, sk = jax.random.split(rng_key)
        policy = RandomPolicy(sk, env.sample_action)
    else:
        policy = MPC(
            action_sample=env.sample_action(PRNGKey(0)),
            cost_fn=env.cost,
            model_fn=curr_model_fn,
            horizon_length=config.traj_length,
            solver=make_solver(),
            receed=False
            #replan=True
        )
    data = EnvDataset(rng_key, env, config.traj_length, policy)[:num_traj]
    data = PyTreeDataset.from_dataset(data)
    logger.info("Generated {} trajectories", data.length)
    traj0 = jax.tree_util.tree_map(lambda x: x[0,...], data.data)
    logger.info("sample traj: states {}", traj0.states)
    logger.info("actions {}", traj0.actions)
    data = data.map(map_fn).flatten()
    # Actually generate the data
    data = PyTreeDataset.from_dataset(data)
    logger.info("Generated {} additional samples", data.length)
    if prev_data is not None:
        data = PyTreeDataset(jax.tree_util.tree_map(lambda x,y: jnp.concatenate((x,y)),
                                            data.data, prev_data.data))
    logger.info("Dataset contains {} samples", data.length)
    return data

def evaluate_model(config, env, est_model_fn, traj_key, gt=False):
    policy = MPC(
        action_sample=env.sample_action(PRNGKey(0)),
        cost_fn=env.cost,
        model_fn=est_model_fn,
        horizon_length=config.traj_length,
        solver=make_solver(gt=gt),
        receed=False
    )
    def eval(key):
        r = stanza.policies.rollout(env.step, env.reset(key), policy,
                                    length=config.traj_length)
        return r, env.cost(r.states, r.actions)
    keys = jax.random.split(traj_key, config.eval_trajs)
    r, c = jax.vmap(eval)(keys)
    r0 = jax.tree_util.tree_map(lambda x: x[0], r)
    logger.info("Eval states: {}", r0.states)
    logger.info("Eval actions: {}", r0.actions)
    return c

def net_fn(config, x, u):
    activation = _ACTIVATION_MAP[config.activation]
    # default network that does not interpret state values
    u_flat, _ = jax.flatten_util.ravel_pytree(u)
    x_flat, unflatten = jax.flatten_util.ravel_pytree(x)
    mlp = hk.nets.MLP(
        [config.hidden_dim]*config.hidden_layers + [x_flat.shape[0]],
        activation=activation
    )
    input = jnp.concatenate((x_flat, u_flat))
    x_flat = mlp(input) + x_flat
    x = unflatten(x_flat)
    return x

def loss_fn(config, net, params, rng_key, sample):
    x, u, x_next = sample
    pred_x = net.apply(params, None, x, u)

    x_next_flat, _ = jax.flatten_util.ravel_pytree(x_next)
    x_pred_flat, _ = jax.flatten_util.ravel_pytree(pred_x)
    loss = jnp.sum(jnp.square(x_next_flat - x_pred_flat))
    return loss, {'loss': loss}

def fit_model(config, net, dataset, rng_key):
    rng = hk.PRNGSequence(rng_key)
    x_sample, u_sample, _ = dataset.get(dataset.start)
    init_params = net.init(next(rng), x_sample, u_sample)

    optimizer = optax.chain(
        # Set the parameters of Adam. Note the learning_rate is not here.
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        optax.additive_weight_decay(0.0001),
        optax.scale_by_schedule(optax.cosine_decay_schedule(1.0,
                                config.iterations)),
        # Put a minus sign to *minimise* the loss.
        optax.scale(-config.lr))

    loss_fn_f = partial(loss_fn, config, net)
    trainer = Trainer(
        loss_fn=loss_fn_f,
        batch_size=config.batch_size,
        max_iterations=config.iterations,
        optimizer=optimizer
    )
    res = trainer.train(dataset, rng_key, init_params,
                        log_interval=1000)
    learned_model = lambda x, u: net.apply(res.fn_params, None, x, u)

    train_loss, train_loss_dict = jax.vmap(partial(loss_fn_f, res.fn_params, None))(dataset.data)
    logger.info("Train Loss {}, {}", jnp.mean(train_loss), jax.tree_map(lambda x: jnp.mean(x), train_loss_dict))
    # test_loss, test_loss_dict = jax.vmap(partial(loss_fn, res.fn_params, None))((xs_eval, us_eval))
    # logger.info("Test Loss {}, {}", jnp.mean(test_loss), jax.tree_map(lambda x: jnp.mean(x), test_loss_dict))
    return learned_model

@activity(Config)
def ilqr_learning(config, db):
    # set the per-env defaults
    if config.env == "pendulum":
        set_default(config, "lr", 1e-3)
        set_default(config, "hidden_dim", 96)
        set_default(config, "activation", "swish")  # might want to sweep over gelu/swish
    elif config.env == "quadrotor":
        set_default(config, "lr", 5e-3)
        set_default(config, "hidden_dim", 128)
        set_default(config, "activation", "gelu")  # might want to sweep over gelu/swish

    env = stanza.envs.create(config.env)
    rng_key = stanza.util.random.key_or_seed(config.rng_seed)
    rng_key, eval_key = jax.random.split(rng_key)

    logger.info("Evaluating optimal cost")
    opt_cost = evaluate_model(config, env, env.step, eval_key, gt=True)
    logger.info("Optimal cost: {}", opt_cost)

    # transform the network to a pure function
    net = hk.transform(partial(net_fn, config))
    est_model_fn = None

    # populate the desired trajectories
    trajectories = [config.init_trajectories]
    while trajectories[-1] < config.total_trajectories:
        t = min(trajectories[-1] + config.traj_interval, config.total_trajectories)
        trajectories.append(t)

    data = None
    total_trajs = 0
    metrics = []
    rng = hk.PRNGSequence(rng_key)
    for t in trajectories:
        logger.info("Running with {} trajectories", t)
        num_trajs = t - total_trajs
        total_trajs = t
        logger.info("Generating data...")
        data = generate_dataset(config, env, est_model_fn, next(rng), num_trajs, data)
        logger.info("Fitting model...")
        est_model_fn = fit_model(config, net, data, next(rng))
        cost = evaluate_model(config, env, est_model_fn, eval_key)
        subopt = (cost - opt_cost)/opt_cost
        subopt_m = jnp.mean(subopt)
        subopt_std = jnp.std(subopt)
        logger.info("Cost {}, suboptimality: {} ({})", cost, subopt_m, subopt_std)
        metrics.append((total_trajs, subopt_m, subopt_std))

    # Make a plot from the metrics
    metrics = jnp.array(metrics)
    fig = px.line(x=metrics[:,0], y=metrics[:,1], log_y=True, error_y=metrics[:,1])
    fig.update_layout(
        xaxis_title="Trajectories",
        yaxis_title="Cost Suboptimality",
        yaxis_range=[-4, 0.5]
    )
    fig.show()