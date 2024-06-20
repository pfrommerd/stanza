from stanza.data import Data, PyTreeData
from stanza.dataclasses import dataclass, field, replace

from typing import Any, Generic, TypeVar

import pickle
import jax.tree_util
import jax.numpy as jnp
import numpy as np

T = TypeVar('T')
I = TypeVar('I')

# For sequences of trajectories,
# use this as the element type
@dataclass
class Step:
    state: jax.Array
    observation: jax.Array
    action: jax.Array

@dataclass
class SequenceInfo(Generic[I]):
    info: I
    start_idx: int
    end_idx: int
    length: int

@dataclass
class Chunk(Generic[T,I]):
    start_offset: int
    chunk: T
    info: I

@dataclass
class ChunkData(Data, Generic[T,I]):
    elements: Data[T]
    sequences: Data[SequenceInfo[I]]
    # contains the timepoints, infos offsets
    # offset by points_offset, infos_offset
    chunk_offsets: Data[tuple[int, int]]
    chunk_length: int = field(pytree_node=False)

    def __len__(self) -> int:
        return len(self.chunk_offsets)
    
    def __getitem__(self, i) -> Chunk[T, I]:
        t_off, i_off = self.chunk_offsets[i]
        info = self.sequences[i_off]
        chunk = self.elements.slice(t_off, self.chunk_length).as_pytree()
        return Chunk(
            start_offset=t_off - info.start_idx,
            chunk=chunk,
            info=info.info
        )

@dataclass
class SequenceData(Generic[T,I]):
    elements: Data[T]
    # contains the start, end, length
    # of each trjaectory
    # in the trajectories data
    sequences: Data[SequenceInfo[I]]

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return self.elements.slice(seq.start_idx, seq.length)
    
    def append(self, data: "SequenceData[T,I]"):
        last_idx = len(self.elements)
        def add_idx(info: SequenceInfo[I]):
            return replace(info,
                start_idx=info.start_idx + last_idx,
                end_idx=info.end_idx + last_idx
            )
        return SequenceData(
            elements=self.elements.append(data.elements),
            sequences=self.sequences.append(data.sequences.map(add_idx))
        )

    def uniform_truncated(self, length: int) -> Data[T]:
        infos = self.sequences.as_pytree()
        mask = infos.length < length
        infos = jax.tree_util.tree_map(lambda x: x[mask], infos)
        start_idxs = infos.start_idxs
        elements = jax.vmap(
            lambda x: self.elements.slice(x, length).as_pytree()
        )(start_idxs)
        return PyTreeData(elements)

    def uniform_repeated(self, length: int) -> Data[T]:
        infos = self.sequences.as_pytree()
        mask = infos.length < length
        def gen_indices(s_index, traj_len):
            return s_index + jnp.minimum(jnp.arange(length), traj_len - 1)
        indices = jax.vmap(gen_indices)(infos.start_idx, infos.length)
        elements = jax.vmap(jax.vmap(lambda x: self.elements[x]))(indices)
        return PyTreeData(elements)
    
    def chunk(self, chunk_length: int, chunk_stride: int = 1) -> ChunkData[T,I]:
        total_chunks = 0
        infos = self.sequences.as_pytree()
        chunks = (infos.length - chunk_length + chunk_stride) // chunk_stride
        chunks = jnp.maximum(0, chunks)
        start_chunks = jnp.cumsum(chunks) - chunks
        total_chunks = jnp.sum(chunks)
        t_off, i_off = np.zeros((2, total_chunks), dtype=jnp.int32)

        for i in range(len(self.sequences)):
            idx = start_chunks[i]
            n_chunks = chunks[i]
            t_off[idx:idx+n_chunks] = infos.start_idx[i] + np.arange(n_chunks) * chunk_stride
            i_off[idx:idx+n_chunks] = i
            idx += n_chunks
        t_off, i_off = jnp.array(t_off), jnp.array(i_off)

        return ChunkData(
            elements=self.elements,
            sequences=self.sequences,
            chunk_offsets=PyTreeData((t_off, i_off)),
            chunk_length=chunk_length
        )
    
    @staticmethod
    def from_trajectory(elements: Data[T], info: I = None) -> "SequenceData[T,I]":
        info = SequenceInfo(
            info=info,
            start_idx=jnp.array(0),
            end_idx=jnp.array(len(elements)),
            length=jnp.array(len(elements))
        )
        sequences = PyTreeData(jax.tree_util.tree_map(lambda x: x[None,...], info))
        return SequenceData(
            elements=elements,
            sequences=sequences
        )