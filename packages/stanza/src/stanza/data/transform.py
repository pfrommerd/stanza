import jax
import jax.numpy as jnp

from typing import Protocol, Generic, TypeVar

T = TypeVar("T")

class Transform(Protocol, Generic[T]):
    def __call__(self, rng_key: jax.Array, x: T) -> T: ...

def random_horizontal_flip(rng_key, x, p=0.5):
    flip = jax.random.bernoulli(rng_key, p, ()) == 1
    return jax.lax.cond(flip, lambda x: jnp.flip(x, axis=-2), lambda x: x, x)

def random_subcrop(rng_key, x, size, padding, padding_mode="constant", padding_constant=0):
    if isinstance(padding, int):
        padding = ((padding,padding),)*(len(x.shape) - 1)
    padding = padding + ((0,0),)
    x = jnp.pad(x, padding, mode=padding_mode, constant_values=padding_constant)
    max_y = x.shape[-3] - size[0]
    max_x = x.shape[-2] - size[1]
    rand_offset = jax.random.randint(rng_key, (2,), 0, jnp.array((max_x, max_y)), dtype=jnp.int32)
    x = jax.lax.dynamic_slice(x,
        [rand_offset[0], rand_offset[0], 0],
        [size[0], size[1], x.shape[-1]]
    )
    return x

# Will black out a (size, size) box with probability p
def random_cutout(rng_key, x, size, p=0.5):
    a, b, c = jax.random.split(rng_key, 3)
    cutout = jax.random.bernoulli(a, p, ()) == 1.
    def do_cutout():
        left, top = jax.random.randint(b, (2,),
            -size//2, jnp.array([x.shape[1] - size//2, x.shape[0] - size//2])
        )
        val = jax.random.uniform(c, ())
        right, bottom = left + size, top + size
        lr_mask = jnp.logical_and(left <= jnp.arange(x.shape[1]), jnp.arange(x.shape[1]) <= right)
        tb_mask = jnp.logical_and(top <= jnp.arange(x.shape[0]), jnp.arange(x.shape[0]) <= bottom)
        mask = lr_mask[None,:] * tb_mask[:, None]
        return jnp.where(mask[...,None], val, x)
    return jax.lax.cond(cutout, do_cutout, lambda: x)