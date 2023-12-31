import math
import math
import functools
import jax
import jax.tree_util as tree_util
import jax.numpy as jnp
import numpy as np

from jax.tree_util import register_pytree_node_class
from stanza.util.logging import logger
from stanza.dataclasses import dataclass, field

from typing import Callable, Any

# We use a float to represent infinite
# datasets so that we can manipulate the
# dataset size and the result is still coherent
INFINITE = jnp.inf
UNKNOWN = None

class Data:
    # Please override the following:
    @property
    def start(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement start"
        )

    def next(self, iterator):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement next"
        )

    # Will move an iterator n forward. By default
    # just calls a for loop, but may be more efficient
    # for some Data sources!
    def advance(self, iterator, n):
        return jax.lax.fori_loop(0, n, lambda i, x: self.next(x), iterator)

    def get(self, iterator):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get"
        )

    # Returns number of elements remainng
    # in the dataset *including* the provided
    # iterator. i.e. if the iterator is at the end
    # of the dataset, this should be 0.
    def remaining(self, iterator):
        return UNKNOWN

    def is_end(self, iterator):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement is_end"
        )
    
    def at(self, idx):
        i = self.start
        i = self.advance(i, idx)
        return self.get(i)

    def scan(self, scan_fn, state, limit=None, jit=True):
        start = self.start
        state = (start, 0, state)
        def cond_fn(state):
            i, n, _ = state
            return jnp.logical_and(jnp.logical_not(self.is_end(i)), 
                    n < limit if limit is not None else True)
        def step_fn(state):
            data = self.get(state[0])
            next_it = self.next(state[0])
            ns = scan_fn(state[2], data)
            return (next_it, state[1] + 1, ns)
        if jit:
            state = jax.lax.while_loop(cond_fn, step_fn, state)
        else:
            while not cond_fn(state):
                state = step_fn(state)
        return state[2]

    # These should NOT be overridden:

    @property
    def length(self):
        return self.remaining(self.start)

    def __iter__(self):
        it = self.start
        while not self.is_end(it):
            o = self.get(it)
            yield o
            it = self.next(it)
    
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if isinstance(i, slice):
            if i.start is not None:
                start = jax.lax.cond(i.start < 0, 
                    lambda: i.start + self.length,
                    lambda: i.start)
            else:
                start = 0
            if i.stop is not None:
                stop = jax.lax.cond(i.stop < 0,
                    lambda: i.stop + self.length,
                    lambda: i.stop)
                l = stop - start
            else:
                l = None
            if i.step is not None:
                raise ValueError("Step must be None")
            start_it = self.advance(self.start, start)
            return self.slice(start_it, l)
        else:
            it = self.advance(self.start, i)
            return self.get(it)

    # ------------------------- Transformations --------------------------------
    def slice(self, start_iter, length):
        return DataSlice(self, start_iter, length)
    
    # Will return the first batch separately.
    # The first batch may be jagged if length % n != 0
    def batch(self, n):
        return BatchedData(self, n)
    
    def flatten(self):
        return FlatData(self)

    def map(self, fun):
        return MappedData(self, fun)

    # Not all datasets implement shuffle!
    def shuffle(self, key):
        raise NotImplementedError("Dataset does not implement shuffle()")
    
    def sample(self, rng_key):
        raise NotImplementedError("Dataset does not implement sample()")

    def sample_batch(self, n, rng_key):
        raise NotImplementedError("Dataset does not implement sample_batch()")
    
    @staticmethod
    def from_pytree(data):
        return PyTreeData(data)

# Type alias for iterators
Iterator = Any

@dataclass(jax=True)
class SliceIterator:
    it: Iterator 
    slice_idx: int

@dataclass(jax=True)
class DataSlice(Data):
    data: Data
    start_iter: Iterator
    n : int

    @property
    def start(self):
        return SliceIterator(self.start_iter, 0)

    def next(self, iterator):
        return SliceIterator(
            self.data.next(iterator.it),
            iterator.slice_idx + 1
        )

    def remaining(self, iterator):
        if self.n is not None:
            dr = self.data.remaining(iterator.it)
            r = self.n - iterator.slice_idx
            # return whichever of dr, r is less
            return jnp.minimum(dr, r)
        else:
            return self.data.remaining(iterator.it)
    
    def is_end(self, iterator):
        if self.n is not None:
            return jnp.logical_or(
                self.data.is_end(iterator.it),
                iterator.slice_idx >= self.n
            )
        else:
            return self.data.is_end(iterator.it)

    def get(self, iterator):
        return self.data.get(iterator.it)

@dataclass(jax=True)
class StaticDataSlice(DataSlice):
    # make the n field static
    n : int = field(jax_static=True)

@dataclass(jax=True)
class PyTreeData(Data):
    data: jnp.array

    @property
    def start(self):
        return 0

    def remaining(self, iter):
        n = tree_util.tree_flatten(self.data)[0][0].shape[0]
        return n - iter
    
    def is_end(self, iter):
        n = tree_util.tree_flatten(self.data)[0][0].shape[0]
        return iter >= n

    def next(self, iter):
        return iter + 1

    def advance(self, iterator, n):
        return iterator + n

    def get(self, iter):
        return tree_util.tree_map(lambda x: x[iter], self.data)
    
    def scan(self, scan_fn, state, limit=None, jit=True):
        if not jit:
            return super().scan(scan_fn, state, limit, jit)
        carry = (0, state)
        def scan(carry, data):
            if limit is not None:
                ns = jax.lax.cond(carry[0] < limit, 
                        scan_fn, lambda x, _: x, carry[1], data)
            else:
                ns = scan_fn(carry[1], data)
            carry = (carry[0] + 1, ns)
            return carry, None
        carry, _ = jax.lax.scan(scan, carry, self.data)
        return carry[1]
    
    def batch(self, n):
        dim = tree_util.tree_flatten(self.data)[0][0].shape[0]
        # get the amount of padding we need
        rem = dim % n
        def make_batches(x):
            if rem > 0:
                x = jax.tree_util.tree_map(lambda x: x[:-rem], x)
            x = x.reshape((-1,n) + x.shape[1:])
            return x
        batched = jax.tree_util.tree_map(
            make_batches, self.data
        )
        return PyTreeData(
            PyTreeData(batched),
        )

    def flatten(self):
        # Add an optimization for PyTreeData
        # of PyTreeData
        if isinstance(self.data, PyTreeData):
            data = jax.tree_util.tree_map(
                lambda x: x.reshape((-1,) + x.shape[2:]),
                self.data.data)
            return PyTreeData(data)
        return super().flatten()

    # This dataset type is shuffleable
    def shuffle(self, key):
        from stanza.util.random import permutation
        dim = tree_util.tree_flatten(self.data)[0][0].shape[0]
        idx = jax.random.permutation(key, dim)
        data = jax.tree_util.tree_map(
            lambda x: jnp.take(x, idx, axis=0, unique_indices=True),
        self.data)
        return PyTreeData(data)
    
    def sample(self, key):
        dim = tree_util.tree_flatten(self.data)[0][0].shape[0]
        idx = jax.random.randint(key, (), 0, dim)
        x = jax.tree_map(lambda x: x[idx], self.data)
        return x
    
    def sample_batch(self, n, rng_key):
        dim = tree_util.tree_flatten(self.data)[0][0].shape[0]
        idx = jax.random.randint(rng_key, (n,), 0, dim)
        x = jax.tree_map(lambda x: x[idx], self.data)
        return x

    @staticmethod
    def from_data(data, start=None, buffer_size=None,
                  fixed_buffer_size=None, chunk_size=None):
        if isinstance(data, PyTreeData) and start is None:
            if buffer_size is not None:
                buf = jax.tree_util.tree_map(
                    lambda x: x[:buffer_size], data.data
                )
                return PyTreeData(buf)
            # If we were supposed to chunk, just
            # read in all of the data anyways...
            return data
        # : If we can calculate the buffer size statically
        # automatically use that
        l = data.length
        if buffer_size is None and fixed_buffer_size is None \
                and l is not None and math.isfinite(l):
            buffer_size = int(l)

        if buffer_size is None and chunk_size is None and fixed_buffer_size is None:
            raise RuntimeError("Must specify buffer or chunk size")

        if start is None:
            start = data.start
        # fixed_buffer_size ignores if the data has less than the buffer
        # size and just fills the whole buffer
        if fixed_buffer_size is not None:
            data, _, _ = PyTreeData._data_chunk(data, start, fixed_buffer_size)
            return PyTreeData(data)
        if buffer_size is not None:
            data, n, _ = PyTreeData._data_chunk(data, start, buffer_size)
            data = jax.tree_util.tree_map(lambda x: x[:n], data)
            return PyTreeData(data)
        else:
            items = []
            n = 0
            iter = start
            while not data.is_end(iter):
                chunk, sn, iter = PyTreeData._data_chunk(data, iter, chunk_size)
                # cut the chunk down to size
                chunk = jax.tree_util.tree_map(lambda x: x[:sn], chunk)
                n = n + sn
                items.append(chunk)
            if len(items) == 0:
                return PyTreeData(None, 0)
            data = jax.tree_util.tree_map(
                lambda *x: jnp.concatenate(x), *items)
            data = jax.tree_util.tree_map(lambda x: x[:n], data)
            return PyTreeData(data)
    
    # Returns
    @staticmethod
    @functools.partial(jax.jit, static_argnums=(2,))
    def _data_chunk(data, start, buffer_size):
        def scan_fn(carry, _):
            iter, n = carry
            new_iter, n = jax.lax.cond(data.is_end(iter),
                lambda: (iter, n),
                lambda: (data.next(iter), n + 1)
            )
            return (new_iter, n), iter
        (iter, n), iters = jax.lax.scan(scan_fn, (start, 0), None,
                    length=buffer_size)
        # in parallel fetch the iterators...
        data = jax.vmap(data.get)(iters)
        return data, n, iter

@dataclass(jax=True)
class MappedData(Data):
    data: Data
    fun: Callable

    @property
    def start(self):
        return self.data.start
    
    def is_end(self, iterator):
        return self.data.is_end(iterator)
    
    def remaining(self, iterator):
        return self.data.remaining(iterator)
    
    @property
    def length(self):
        return self.data.length

    def next(self, iterator):
        return self.data.next(iterator)

    def get(self, iterator):
        item = self.data.get(iterator)
        return self.fun(item)
    
    # override slice, shuffle, batch transformations
    # to happen pre-map application since this is 
    # generally more efficient (i.e for PyTreeData)
    def slice(self, start_iter, len):
        return MappedData(self.data.slice(start_iter, len), self.fun)

    def batch(self, n):
        # apply the map function to each batch object
        def map_batch(x):
            return x.map(self.fun)
        return MappedData(self.data.batch(n), map_batch)
    
    def shuffle(self, rng_key):
        return MappedData(self.data.shuffle(rng_key), self.fun)


# n should not be part
# of the jax tree
@dataclass(jax=True)
class BatchedData(Data):
    data: Data
    n : int = field(jax_static=True)

    @property
    def start(self):
        return self.data.start
    
    def is_end(self, iterator):
        return self.remaining(iterator) == 0

    def remaining(self, iterator):
        return self.data.remaining(iterator) // self.n 

    def get(self, iterator):
        return StaticDataSlice(self.data, iterator, self.n)

    def next(self, iterator):
        # advance the iterator by n
        return self.data.advance(iterator, self.n)

    def flatten(self):
        return self.data

@dataclass(jax=True)
class FlatIterator:
    it: Iterator
    sub_it: Iterator

@dataclass(jax=True)
class FlatData(Data):
    data: Data

    @property
    def start(self):
        upper_it = self.data.start
        upper_data = self.data.get(upper_it)
        return FlatIterator(upper_it, upper_data.start)
    
    def is_end(self, iterator):
        return self.data.is_end(iterator.it)

    def next(self, iterator):
        sub_data = self.data.get(iterator.it)
        sit = sub_data.next(iterator.sub_it)
        def advance_main():
            # advance it, get new sub_it
            new_it = self.data.next(iterator.it)
            sub_data = self.data.get(new_it)
            new_sub_it = sub_data.start
            return FlatIterator(new_it, new_sub_it)
        return jax.lax.cond(
            sub_data.is_end(sit),
            advance_main,
            lambda: FlatIterator(iterator.it, sit)
        )

    def get(self, iterator):
        sub_data = self.data.get(iterator.it)
        return sub_data.get(iterator.sub_it)