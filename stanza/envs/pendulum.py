from stanza.envs import Environment

import jax
import jax.numpy as jnp
import numpy as np
import math

from typing import NamedTuple
from functools import partial
from stanza.runtime.database import Figure, Video

import math
import plotly.express as px

class State(NamedTuple):
    angle: jnp.ndarray
    vel: jnp.ndarray

class PendulumEnv(Environment):
    def __init__(self, sub_steps=1):
        self.sub_steps = sub_steps
        self.dt = 0.1

    def sample_action(self, rng_key):
        return jax.random.uniform(
            rng_key, shape=(), minval=-1.0, maxval=1.0)

    # Sample state should be like reset(),
    # but whereas reset() is meant to be a distribution
    # over initial conditions, sample_state() should
    # give a distribution with support over all possible (or reasonable) states
    def sample_state(self, rng_key):
        k1, k2 = jax.random.split(rng_key)
        angle = 5*jax.random.uniform(k1, shape=(), minval=-1, maxval=math.pi + 1)
        vel = 5*jax.random.uniform(k2, shape=(), minval=-1, maxval=1)
        return State(angle, vel)

    def reset(self, key):
        # pick random position between +/- radians from center
        angle = jax.random.uniform(key,shape=(), minval=-1,maxval=+1)
        vel = jnp.zeros(())
        return State(angle, vel)

    def step(self, state, action):
        angle = state.angle + self.dt*state.vel
        vel = state.vel + self.dt*jnp.sin(state.angle) + self.dt*action
        state = State(angle, vel)
        return state
    
    # If u is None, this is the terminal cost
    def cost(self, x, u):
        x = jnp.stack((x.angle, x.vel), -1)
        diff = (x - jnp.array([math.pi, 0]))
        x_cost = jnp.sum(diff**2)
        xf_cost = jnp.sum(diff[-1]**2)
        u_cost = jnp.sum(u**2)
        return 100*xf_cost + x_cost + 0.1*u_cost

    def constraints(self, _, us):
        constraints = [jnp.ravel(us - 3),
                       jnp.ravel(-3 - us)]
        return jnp.concatenate(constraints)

    def visualize(self, states, actions):
        traj = px.line(x=jnp.squeeze(states.angle, -1), y=jnp.squeeze(states.vel, -1))
        traj.update_layout(xaxis_title="Theta", yaxis_title="Omega", title="State Trajectory")

        theta = px.line(x=jnp.arange(states.angle.shape[0]), y=jnp.squeeze(states.angle, -1))
        theta.update_layout(xaxis_title="Time", yaxis_title="Theta", title="Angle")

        omega = px.line(x=jnp.arange(states.vel.shape[0]), y=jnp.squeeze(states.vel, -1))
        omega.update_layout(xaxis_title="Time", yaxis_title="Omega", title="Angular Velocity")

        u = px.line(x=jnp.arange(actions.shape[0]), y=jnp.squeeze(actions, -1))
        u.update_layout(xaxis_title="Time", yaxis_title="u")

        video = jax.vmap(self.render)(states)

        return {
            'video': Video(video, fps=15),
            'traj': Figure(traj),
            'theta': Figure(theta),
            'omega': Figure(omega),
            'u': Figure(u)
        }

    def render(self, state, width=256, height=256):
        return jax.pure_callback(
            partial(render_pendulum, width=width, height=height),
            jax.ShapeDtypeStruct((3, width, height), jnp.uint8),
            state
        )


def render_pendulum(state, width, height):
    from cairo import ImageSurface, Context, Format
    surface = ImageSurface(Format.ARGB32, width, height)
    ctx = Context(surface)
    ctx.rectangle(0, 0, width, height)
    ctx.set_source_rgb(0.9, 0.9, 0.9)
    ctx.fill()
    ctx.move_to(width/2, height/2)

    radius = 0.7*min(width, height)/2
    ball_radius = 0.1*min(width, height)/2

    # put theta through a tanh to prevent
    # wraparound
    theta = state.angle + math.pi

    x = np.sin(theta)*radius + width/2
    y = np.cos(theta)*radius + height/2

    ctx.set_source_rgb(0.1, 0.1, 0.1)
    ctx.set_line_width(1)
    ctx.line_to(x, y)
    ctx.stroke()

    ctx.set_source_rgb(0.9, 0, 0)
    ctx.arc(x, y, ball_radius, 0, 2*math.pi)
    ctx.fill()
    img = cairo_to_numpy(surface)[:3,:,:]
    # we need to make a copy otherwise it may
    # get overridden the next time we render
    return np.copy(img)

def cairo_to_numpy(surface):
    data = np.ndarray(shape=(surface.get_height(), surface.get_width(), 4),
                    dtype=np.uint8,
                    buffer=surface.get_data())
    data[:,:,[0,1,2,3]] = data[:,:,[2,1,0,3]]
    data = np.transpose(data, (2, 0, 1))
    return data

def builder(name):
    return PendulumEnv()