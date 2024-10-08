from foundry.core.dataclasses import dataclass, field
from foundry.solver import UnsupportedObectiveError, MinimizeState, Minimize
from foundry.solver.iterative import IterativeSolver

from typing import Any
import jax
import foundry.numpy as jnp

import optax
import jax
import foundry.numpy as jnp


@dataclass
class OptaxState(MinimizeState):
    optimizer_state : Any

@dataclass(kw_only=True)
class OptaxSolver(IterativeSolver):
    tol: float = 1e-3
    optimizer: Any = None

    @jax.jit
    def init(self, objective):
        if not isinstance(objective, Minimize):
            raise UnsupportedObectiveError("Can only handle unconstrained minimization objectives")
        return OptaxState(
            iteration=jnp.zeros((), dtype=jnp.int32),
            solved=jnp.array(False),
            state=objective.initial_state,
            params=objective.initial_params,
            cost=jnp.zeros(()),
            aux=None,
            optimizer_state = None #self.optimizer.init(objective.initial_params)
        )

    @jax.jit
    def optimality(self, objective, solver_state):
        grad = jax.grad(lambda p: objective.eval(solver_state.state, p)[1])(solver_state.params)
        return grad

    @jax.jit
    def update(self, objective, solver_state):
        return solver_state
        if not isinstance(objective, Minimize) or objective.constraints:
            raise UnsupportedObectiveError("Can only handle unconstrained minimization objectives")
        def f(p):
            obj_state, cost, obj_aux = objective.eval(solver_state.state, p)
            return cost, (obj_state, cost, obj_aux)
        grad, (obj_state, cost, obj_aux) = jax.grad(f, has_aux=True)(solver_state.params)
        updates, new_opt_state = self.optimizer.update(grad, solver_state.optimizer_state, solver_state.params)
        obj_params = optax.apply_updates(solver_state.params, updates)
        return OptaxState(
                    iteration=solver_state.iteration + 1,
                    solved=jnp.array(False),
                    state=obj_state,
                    params=obj_params,
                    cost=cost,
                    aux=obj_aux,
                    optimizer_state=new_opt_state
                )