# A simulator which uses pure_callback
# and a cache of SytemState -> MjData

from ..core import Simulator, SystemState, SystemData

from stanza.dataclasses import dataclass, replace
from concurrent.futures.thread import ThreadPoolExecutor

from collections import OrderedDict

import mujoco
import jax
import jax.numpy as jnp
import numpy as np
import os
import threading


# the mujoco simulator state is
# the SystemData and qacc_warmstart
@dataclass
class MujocoState:
    data: SystemData
    qacc_warmstart: jax.Array

# Used for the host <-> accelerator calls, the
# minimum state that needs to be passed to mimic simulator state
@dataclass
class MujocoStep:
    time: jax.Array
    qpos: jax.Array
    qvel: jax.Array
    act: jax.Array
    ctrl: jax.Array
    qacc_warmstart: jax.Array

class MujocoSimulator(Simulator[SystemData]):
    def __init__(self, model, threads=os.cpu_count()):
        # store up to 256 MjData objects for this model
        self.model = model
        example_data = mujoco.MjData(self.model)
        example_data = SystemData(np.array(example_data.time),
            example_data.qpos, example_data.qvel, 
            example_data.act, example_data.qacc, example_data.act_dot,
            example_data.xpos, example_data.xquat,
            example_data.actuator_velocity, example_data.cvel
        )
        self.data_structure = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape,
                x.dtype if x.dtype != np.float64 else np.float32
            ),
            example_data
        )

        self.local_data = threading.local()
        # create an initial MjData object for each thread
        def initializer():
            self.local_data.data = mujoco.MjData(self.model)
        self.pool = ThreadPoolExecutor(
            max_workers=threads, 
            initializer=initializer
        )

    def _step_job(self, step: MujocoStep) -> SystemData:
        # get the thread-local MjData object
        # copy over the jax arrays
        data = self.local_data.data
        data.time = step.time.item()
        data.qpos[:] = step.qpos
        data.qvel[:] = step.qvel
        data.act[:] = step.act
        data.ctrl[:] = step.ctrl
        data.qacc_warmstart[:] = step.qacc_warmstart
        mujoco.mj_step(self.model, data)
        return MujocoState(
            data=SystemData(jnp.array(data.time), 
                jnp.copy(data.qpos), jnp.copy(data.qvel), 
                jnp.copy(data.act), jnp.copy(data.qacc), jnp.copy(data.act_dot),
                jnp.copy(data.xpos), jnp.copy(data.xquat),
                jnp.copy(data.actuator_velocity), jnp.copy(data.cvel)
            ),
            qacc_warmstart=jnp.copy(data.qacc_warmstart)
        )

    # (on host) step using the minimal amount of 
    # data that needs to be passed to the simulator
    def _step(self, step: MujocoStep) -> SystemData:
        if step.qpos.ndim == 1:
            data = self.pool.submit(self._step_job, step).result()
            return data
        else:
            batch_shape = step.qpos.shape[:-1]
            batch_size = np.prod(batch_shape)
            step = jax.tree.map(
                lambda x: jnp.reshape(x, (batch_size,) + x.shape[len(batch_shape):]),
                step
            )
            # split the step batch into a list
            steps = [jax.tree.map(lambda x: x[i], step) for i in range(step.qpos.shape[0])]
            # submit to the thread pool
            data = self.pool.map(self._step_job, steps)
            # stack the results and return
            data = jax.tree.map(lambda *x: jnp.stack(x), *data)
            # reshape the data back to the original shape
            data = jax.tree.map(
                lambda x: jnp.reshape(x, batch_shape + x.shape[1:]),
                data
            )
            return data

    def step(self, state: MujocoState,
                   action : jax.Array, rng_key: jax.Array) -> SystemData:
        assert state.data.time.ndim == 0
        return jax.pure_callback(
            self._step, MujocoState(
                self.data_structure,
                jax.ShapeDtypeStruct(
                    state.data.qpos.shape, state.data.qpos.dtype
                )
            ), 
            MujocoStep(
                state.data.time, state.data.qpos, state.data.qvel, 
                state.data.act, action, state.qacc_warmstart
            ),
            vectorized=True
        )

    def _forward_job(self, step: MujocoStep) -> MujocoState:
        data = self.local_data.data
        data.time = step.time.item()
        data.qpos[:] = step.qpos
        data.qvel[:] = step.qvel
        data.act[:] = step.act
        if step.ctrl is not None:
            data.ctrl[:] = np.zeros_like(step.ctrl)
        if step.qacc_warmstart is not None:
            data.qacc_warmstart[:] = step.qacc_warmstart
        mujoco.mj_forward(self.model, data)
        return MujocoState(
            data=SystemData(jnp.array(data.time), jnp.copy(data.qpos), jnp.copy(data.qvel), 
                jnp.copy(data.act), jnp.copy(data.qacc), jnp.copy(data.act_dot),
                jnp.copy(data.xpos), jnp.copy(data.xquat),
                jnp.copy(data.actuator_velocity), jnp.copy(data.cvel)
            ),
            qacc_warmstart=jnp.copy(data.qacc_warmstart)
        )

    # (on host) calls forward
    def _forward(self, step: MujocoStep) -> MujocoState:
        if step.qpos.ndim == 1:
            data = self.pool.submit(self._forward_job, step).result()
            return data
        else:
            batch_shape = step.qpos.shape[:-1]
            batch_size = np.prod(batch_shape)
            step = jax.tree.map(
                lambda x: jnp.reshape(x, (batch_size,) + x.shape[len(batch_shape):]),
                step
            )
            steps = [jax.tree.map(lambda x: x[i], step) for i in range(step.qpos.shape[0])]
            data = self.pool.map(self._forward_job, steps)
            data = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *data)
            data = jax.tree.map(
                lambda x: jnp.reshape(x, batch_shape +  x.shape[1:]),
                data
            )
            return data

    def full_state(self, state: SystemState) -> MujocoState:
        assert state.time.ndim == 0
        step = MujocoStep(
            state.time, state.qpos, 
            state.qvel, state.act, None, None
        )
        structure = MujocoState(
            self.data_structure,
            jax.ShapeDtypeStruct(self.data_structure.qpos.shape, 
                                    self.data_structure.qpos.dtype)
        )
        return jax.pure_callback(self._forward, 
            structure, step, vectorized=True)

    def reduce_state(self, state: MujocoState) -> SystemState:
        return SystemState(
            state.data.time, state.data.qpos, state.data.qvel, state.data.act
        )

    def data(self, state: MujocoState) -> SystemData:
        return state.data

jax.tree_util.register_static(MujocoSimulator)