from stanza.env import (
    EnvWrapper, EnvironmentRegistry,
    RenderConfig, ImageRender,
    ObserveConfig
)
from . import assets

from stanza.policy.transforms import Transform, chain_transforms
from stanza.policy import Policy
from stanza.dataclasses import dataclass, field, replace
from stanza.util import jax_static_property
from stanza import canvas

from stanza.env.mujoco.core import (
    MujocoEnvironment, SystemState, 
    SimulatorState, Action,
    quat_to_angle, render_2d
)

import shapely.geometry as sg
import jax.numpy as jnp
import jax.random

import importlib.resources as resources

@dataclass
class PushTObs:
    agent_pos: jnp.array
    agent_vel: jnp.array

    block_pos: jnp.array
    block_vel: jnp.array

    block_rot: jnp.array
    block_rot_vel: jnp.array

@dataclass
class PushTEnv(MujocoEnvironment[SimulatorState]):
    success_threshold: float = field(default=0.9, pytree_node=False)

    goal_pos: jnp.array = field(default_factory=lambda: jnp.array([0, 0]), pytree_node=False)
    goal_rot: jnp.array = field(default_factory=lambda: jnp.array(-jnp.pi/4), pytree_node=False)

    agent_radius : float = field(default=15/252, pytree_node=False)
    block_scale : float = field(default=30/252, pytree_node=False)
    world_scale : float = field(default=2, pytree_node=False)

    @jax_static_property
    def xml(self):
        with (resources.files(assets) / "pusht.xml").open("r") as f:
            xml = f.read()
        com = 0.5*(self.block_scale/2) + 0.5*(self.block_scale + 1.5*self.block_scale)
        return xml.format(
            goal_pos_x=self.goal_pos[0], goal_pos_y=self.goal_pos[1], 
            goal_rot=self.goal_rot, 
            agent_radius=self.agent_radius,

            world_scale=self.world_scale,
            half_world_scale=self.world_scale/2,
            # the other constants needed for the block
            block_scale=self.block_scale,
            half_block_scale=self.block_scale/2,
            double_block_scale=2*self.block_scale,
            one_and_half_block_scale=1.5*self.block_scale,
            two_and_half_block_scale=2.5*self.block_scale,
            com_offset=com
        )

    @jax.jit
    def system_reset(self, rng_key) -> SystemState:
        a_pos, b_pos, b_rot, c = jax.random.split(rng_key, 4)
        agent_pos = jax.random.uniform(a_pos, (2,), minval=-0.8, maxval=0.8)
        block_rot = jax.random.uniform(b_pos, (), minval=-jnp.pi, maxval=jnp.pi)
        block_pos = jax.random.uniform(b_rot, (2,), minval=-0.4, maxval=0.4)
        # re-generate block positions while the block is too close to the agent
        min_radius = self.block_scale*2*jnp.sqrt(2) + self.agent_radius
        def gen_pos(carry):
            rng_key, _ = carry
            rng_key, sk = jax.random.split(rng_key)
            return (rng_key, jax.random.uniform(sk, (2,), minval=-0.4, maxval=0.4))
        _, block_pos = jax.lax.while_loop(
            lambda s: jnp.linalg.norm(s[1] - agent_pos) < min_radius,
            gen_pos, (c, block_pos)
        )
        qpos = jnp.concatenate([agent_pos, block_pos, block_rot[jnp.newaxis]])
        return SystemState(
            jnp.zeros(()), qpos, jnp.zeros_like(qpos), jnp.zeros((0,)))
    
    @jax.jit
    def observe(self, state, config : ObserveConfig = None):
        data = self.simulator.data(state)
        return PushTObs(
            # Extract agent pos, vel
            agent_pos=data.xpos[1,:2],
            agent_vel=data.cvel[1,3:5],
            # Extract block pos, vel, angle, angular vel
            block_pos=data.xpos[2,:2],
            block_rot=quat_to_angle(data.xquat[2,:4]),
            block_vel=data.cvel[2,3:5],
            block_rot_vel=data.cvel[2,2],
        )

    # For computing the reward
    def _block_points(self, pos, rot):
        center_a, hs_a = jnp.array([0, -self.block_scale/2]), \
                jnp.array([2*self.block_scale, self.block_scale/2])
        center_b, hs_b = jnp.array([0, -2.5*self.block_scale]), \
                        jnp.array([self.block_scale/2, 1.5*self.block_scale])

        points = jnp.array([
            center_a + jnp.array([hs_a[0], -hs_a[1]]),
            center_a + hs_a,
            center_a + jnp.array([-hs_a[0], hs_a[1]]),
            center_a - hs_a,
            center_b + jnp.array([-hs_b[0], hs_b[1]]),
            center_b - hs_b,
            center_b + jnp.array([hs_b[0], -hs_b[1]]),
            center_b + hs_b
        ])
        rotM = jnp.array([
            [jnp.cos(rot), -jnp.sin(rot)],
            [jnp.sin(rot), jnp.cos(rot)]
        ])
        points = jax.vmap(lambda v: rotM @ v)(points)
        return points + pos

    @staticmethod
    def _overlap(pointsA, pointsB):
        polyA = sg.Polygon(pointsA)
        polyB = sg.Polygon(pointsB)
        return polyA.intersection(polyB).area / polyA.area

    @jax.jit
    def reward(self, state : SimulatorState, 
                action : Action, 
                next_state : SimulatorState):
        obs = self.observe(next_state)
        goal_points = self._block_points(self.goal_pos, self.goal_rot)
        points = self._block_points(obs.block_pos, obs.block_rot)
        overlap = jax.pure_callback(
            PushTEnv._overlap,
            jax.ShapeDtypeStruct((), jnp.float32),
            goal_points, points
        )
        return jnp.minimum(overlap, self.success_threshold) / self.success_threshold

    @jax.jit
    def render(self, state : SimulatorState, config : RenderConfig | None = None): 
        config = config or ImageRender(width=256, height=256)
        if isinstance(config, ImageRender):
            data = self.simulator.data(state)
            image = jnp.ones((config.height, config.width, 3))
            target = render_2d(
                self.model, data, 
                config.width, config.height,
                2, 2,
                body_custom={2: (self.goal_pos, self.goal_rot, jnp.array([0, 1, 0]))}
            )
            world = render_2d(
                self.model, data, 
                config.width, config.height,
                2, 2
            )
            image = canvas.paint(image, target, world)
            return image
        else:
            raise ValueError("Unsupported render config")


@dataclass
class PushTPosObs:
    agent_pos: jnp.array
    block_pos: jnp.array
    block_rot: jnp.array

@dataclass
class PushTKeypointObs:
    agent_pos: jnp.array
    block_pos: jnp.array
    block_end: jnp.array

@dataclass
class PushTKeypointRelObs:
    agent_block_pos: jnp.array
    agent_block_end: jnp.array
    rel_block_pos: jnp.array
    rel_block_end: jnp.array
        
# A state-feedback adapter for the PushT environment
# Will run a PID controller under the hood
@dataclass
class PositionalControlTransform(Transform):
    k_p : float = 15
    k_v : float = 2

    def transform_policy(self, policy):
        return PositionalControlPolicy(policy, self.k_p, self.k_v)
    
    def transform_env(self, env):
        return PositionalControlEnv(env, self.k_p, self.k_v)

@dataclass
class PositionalControlPolicy(Policy):
    policy: Policy
    k_p : float = 20
    k_v : float = 2

    @property
    def rollout_length(self):
        return self.policy.rollout_length

    def __call__(self, input):
        output = self.policy(input)
        if output.action is None:
            return replace(output, action=jnp.zeros((2,)))
        a = self.k_p * (output.action - output.agent.position) + self.k_v * (-output.agent.velocity)
        return replace(
            output, action=a
        )

@dataclass
class PositionalControlEnv(EnvWrapper):
    k_p : float = 50
    k_v : float = 2

    def step(self, state, action, rng_key=None):
        obs = PushTEnv.observe(self.base, state)
        if action is not None:
            a = self.k_p * (action - obs.agent_pos) + self.k_v * (-obs.agent_vel)
        else: 
            a = jnp.zeros((2,))
        return self.base.step(state, a, None)

@dataclass
class PositionalObsTransform(Transform):
    def transform_policy(self, policy):
        return PositionalObsPolicy(policy)
    
    def transform_env(self, env):
        return PositionalObsEnv(env)

@dataclass
class PositionalObsPolicy(Policy):
    policy: Policy

    @property
    def rollout_length(self):
        return self.policy.rollout_length

    def __call__(self, input):
        obs = input.observation
        obs = PushTPosObs(
            agent_pos=obs.agent.position,
            block_pos=obs.block.position,
            block_rot=obs.block.rotation
        )
        input = replace(input, observation=obs)
        return self.policy(input)

@dataclass
class PositionalObsEnv(EnvWrapper):
    def observe(self, state, config=None):
        obs = self.base.observe(state, config)
        return PushTPosObs(
            agent_pos=obs.agent_pos,
            block_pos=obs.block_pos,
            block_rot=obs.block_rot
        )

@dataclass
class KeypointObsTransform(Transform):
    def transform_policy(self, policy):
        raise NotImplementedError()
    
    def transform_env(self, env):
        return KeypointObsEnv(env)

@dataclass
class KeypointObsEnv(EnvWrapper):
    def observe(self, state):
        obs = self.base.observe(state)
        rotM = jnp.array([
            [jnp.cos(obs.block_rot), -jnp.sin(obs.block_rot)],
            [jnp.sin(obs.block_rot), jnp.cos(obs.block_rot)]
        ])
        end = rotM @ jnp.array([0, -4*self.block_scale]) + obs.block_pos
        return PushTKeypointObs(
            agent_pos=obs.agent_pos,
            block_pos=obs.block_pos,
            block_end=end
        )

@dataclass
class RelKeypointObsTransform(Transform):
    def transform_policy(self, policy):
        raise NotImplementedError()
    
    def transform_env(self, env):
        return RelKeypointObsEnv(env)

@dataclass
class RelKeypointObsEnv(EnvWrapper):
    def observe(self, state, config=None):
        obs = self.base.observe(state)
        rotM = jnp.array([
            [jnp.cos(obs.block_rot), -jnp.sin(obs.block_rot)],
            [jnp.sin(obs.block_rot), jnp.cos(obs.block_rot)]
        ])
        end = rotM @ jnp.array([0, -4*self.block_scale]) + obs.block_pos

        rotM = jnp.array([
            [jnp.cos(self.goal_rot), -jnp.sin(self.goal_rot)],
            [jnp.sin(self.goal_rot), jnp.cos(self.goal_rot)]
        ])
        goal_end = rotM @ jnp.array([0, -4*self.block_scale]) + self.goal_pos
        return PushTKeypointRelObs(
            agent_block_pos=obs.agent_pos - obs.block_pos,
            agent_block_end=obs.agent_pos - end,
            rel_block_pos=obs.block_pos - self.goal_pos,
            rel_block_end=end - goal_end,
        )

environments = EnvironmentRegistry[PushTEnv]()
environments.register(PushTEnv)

def _make_positional(**kwargs):
    env = PushTEnv(**kwargs)
    return chain_transforms(
        PositionalControlTransform(),
        PositionalObsTransform
    ).transform_env(env)
environments.register("positional", _make_positional)

def _make_keypoint(**kwargs):
    env = PushTEnv(**kwargs)
    return chain_transforms(
        PositionalControlTransform(),
        KeypointObsTransform()
    ).transform_env(env)
environments.register("keypoint", _make_keypoint)

def _make_rel_keypoint(**kwargs):
    env = PushTEnv(**kwargs)
    return chain_transforms(
        PositionalControlTransform(),
        RelKeypointObsTransform()
    ).transform_env(env)
environments.register("rel_keypoint", _make_rel_keypoint)