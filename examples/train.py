import jax
from stanza.data import Data
from stanza.util.logging import logger
from jax.random import PRNGKey
import jax.numpy as jnp
import logging
from rich.logging import RichHandler
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.ERROR, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

# A dataset of integers
dataset = Data.from_pytree(
    (jnp.arange(100), jnp.arange(100)[::-1])
)
logger.info("Dataset length: {}", dataset.length)
batches = dataset.batch(20)
logger.info("Batched length: {}", batches.length)

import haiku as hk

def net_fn(input):
    logger.info("Tracing model", only_tracing=True)
    input = jnp.atleast_1d(input)
    y = hk.nets.MLP([10, 1])(input)
    return jnp.squeeze(y, -1)

net = hk.transform(net_fn)
orig_init_params = net.init(PRNGKey(7), jnp.ones(()))

import optax
from stanza.util.random import permutation
import jax

logger.info("Permutation test: {}", permutation(PRNGKey(42), 10, n=6))

optimizer = optax.chain(
    # Set the parameters of Adam. Note the learning_rate is not here.
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
    # Put a minus sign to *minimise* the loss.
    optax.scale(-5e-3),
    optax.scale_by_schedule(optax.cosine_decay_schedule(1.0,
                            5000*10, alpha=0.1))
)

def net_apply(params, rng_key, x):
    params = {m: {k: jnp.array(v) for (k,v) in sp.items()}
                for (m,sp) in params.items()}
    return net.apply(params, rng_key, x)

logger.info("Testing JIT apply")
jit_apply = jax.jit(net_apply)
jit_apply(orig_init_params, None, jnp.ones(()))
jit_apply(orig_init_params, None, jnp.ones(()))
jit_apply(orig_init_params, None, jnp.ones(()))
logger.info("Done testing JIT apply")

def loss_fn(params, rng_key, sample):
    x, y = sample
    out = jit_apply(params, rng_key, x)
    loss = jnp.square(out - y)
    stats = {
        "loss": loss
    }
    return loss, stats

from stanza import Partial
from stanza.train import Trainer
from stanza.train.rich import RichReporter
from stanza.train.wandb import WandbReporter
# import wandb
# wandb.init(project="train_test")

with WandbReporter() as wb:
    with RichReporter(iter_interval=500) as cb:
        trainer = Trainer(epochs=5000, batch_size=10)
        init_params = net.init(PRNGKey(7), jnp.ones(()))
        res = trainer.train(
            Partial(loss_fn), dataset,
            PRNGKey(42), init_params,
            hooks=[cb], jit=True
        )

from stanza.train import _train_jit

logger.info("Train cache size {}", _train_jit._cache_size())

logger.info("Training again...jit is cached so now training is fast")
with WandbReporter() as wb:
    with RichReporter(iter_interval=500) as cb:
        trainer = Trainer(epochs=5000, batch_size=10)
        init_params = net.init(PRNGKey(7), jnp.ones(()))
        res = trainer.train(
            Partial(loss_fn), dataset,
            PRNGKey(42), init_params,
            hooks=[cb], jit=True
        )

logger.info("Training again...but this time without a full JIT loop (sloooow)")
with WandbReporter() as wb:
    with RichReporter(iter_interval=500) as cb:
        trainer = Trainer(epochs=5000, batch_size=10)
        init_params = net.init(PRNGKey(7), jnp.ones(()))
        res = trainer.train(
            Partial(loss_fn), dataset,
            PRNGKey(42), init_params,
            hooks=[cb], jit=False
        )
logger.info("Train cache size {}", _train_jit._cache_size())