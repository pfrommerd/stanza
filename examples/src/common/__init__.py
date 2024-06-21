from stanza import struct
from stanza.config import ConfigProvider

import optax
import jax
import jax.numpy as jnp

@struct.dataclass
class OptimizerConfig:
    lr: float = 1e-4
    lr_schedule: str = "constant"
    cycles: int = 1
    # length to multiply the number of
    # iterations per cycle, every cycle
    cycle_mult: float = 1.
    warmup_schedule: str | None = None
    # if none, will use a "good" value based on warmup_schedule and the number of iterations 
    warmup_steps: int | None = None 

    def make_lr_schedule(self, iterations):
        warmup_steps = (
            100
            if self.warmup_steps is None  and self.warmup_schedule is not None 
            else (self.warmup_steps or 0)
        )
        if self.warmup_schedule is None:
            warmup = None
        elif self.warmup_schedule == "linear":
            warmup = optax.linear_schedule(0, self.lr, warmup_steps)
        elif self.warmup_schedule == "cosine":
            warmup = optax.cosine_decay_schedule(0, warmup_steps, self.lr)
        else:
            raise ValueError(f"Unknown warmup schedule: {self.warmup_schedule}")
        if warmup is not None:
            iterations = iterations - warmup_steps

        if self.lr_schedule == "constant":
            schedule_builder = lambda s: optax.constant_schedule(self.lr)
        elif self.lr_schedule == "linear":
            schedule_builder = lambda s: optax.linear_schedule(self.lr, 0., s)
        elif self.lr_schedule == "cosine":
            schedule_builder = lambda s: optax.cosine_decay_schedule(self.lr, s)
        elif self.lr_schedule == "exponential":
            schedule_builder = lambda s: optax.exponential_decay(
                init_value=self.lr, 
                transition_steps=s/10,
                staircase=False,
                decay_rate=0.8, 
                end_value=0.001*self.lr)
        elif self.lr_schedule == "exponential_staircase":
            schedule_builder = lambda s: optax.exponential_decay(
                init_value=self.lr, 
                transition_steps=s/10,
                staircase=True,
                decay_rate=0.8, 
                end_value=0.001*self.lr)
        else:
            raise ValueError(f"Unknown learning rate schedule: {self.lr_schedule}")

        if self.cycles > 1:
            # total length is base*(1 + m + m^2 ... + m^(cycles - 1)) = 1/(1 - m)
            if self.cycle_mult == 1.:
                base_units = self.cycles
            else:
                base_units = 1/(1 - self.cycle_mult) - (self.cycle_mult ** self.cycles) / (1 - self.cycle_mult)
            base_steps = iterations // base_units
            schedules = []
            boundaries = []
            for i in range(self.cycles):
                part_iterations = base_steps * (self.cycle_mult ** i)
                schedules.append(schedule_builder(part_iterations))
                if i == 1: boundaries.append(base_steps)
                elif i > 1: boundaries.append(boundaries[-1] + part_iterations)
            schedule = optax.join_schedules(schedules, boundaries)
        else:
            schedule = schedule_builder(iterations)
        if warmup is None:
            return schedule
        else:
            return optax.join_schedules([warmup, schedule], [warmup_steps])

    def make_optimizer(self, iterations):
        raise NotImplementedError()

    @staticmethod
    def default():
        return AdamConfig(lr_schedule="cosine", warmup_schedule="linear")

@struct.dataclass
class AdamConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float | None = 0.0001

    def make_optimizer(self, iterations):
        if self.weight_decay is not None:
            return optax.adamw(learning_rate=self.make_lr_schedule(iterations),
                            b1=self.beta1, b2=self.beta2, eps=self.epsilon,
                            weight_decay=self.weight_decay)
        else:
            return optax.adam(learning_rate=self.make_lr_schedule(iterations), 
                            b1=self.beta1, b2=self.beta2, eps=self.epsilon)

    def parse(self, config: ConfigProvider) -> "AdamConfig":
        return config.get_struct(self)

@struct.dataclass
class SGDConfig(OptimizerConfig):
    lr_schedule: str = "constant" # The learning rate schedule
    momentum: float | None = None
    nesterov: bool = False
    weight_decay: float | None = None

    def make_optimizer(self, iterations):
        sgd = optax.sgd(learning_rate=self.make_lr_schedule(iterations),
                         momentum=self.momentum, nesterov=self.nesterov)
        if self.weight_decay:
            sgd = optax.chain(optax.add_decayed_weights(self.weight_decay), sgd)
        return sgd

    def parse(self, config: ConfigProvider) -> "SGDConfig":
        return config.get_struct(self)

from optax._src import base
from optax._src.transform import ScaleByScheduleState, numerics

def combine(
    *args: base.GradientTransformation,
) -> base.GradientTransformationExtraArgs:
    transforms = [base.with_extra_args_support(t) for t in args]
    init_fns, update_fns = zip(*transforms)
    def init_fn(params):
        return tuple(fn(params) for fn in init_fns)
    def update_fn(updates, state, params=None, **extra_args):
        if len(update_fns) != len(state):
            raise ValueError('The number of updates and states has to be the same in '
                           'chain! Make sure you have called init first!')
        new_state = []
        new_updates = []
        for s, fn in zip(state, update_fns):
            updates_out, new_s = fn(updates, s, params, **extra_args)
            new_updates.append(updates_out)
            new_state.append(new_s)
        return tuple(new_updates), tuple(new_state)
    return base.GradientTransformationExtraArgs(init_fn, update_fn)

def update_by_schedule(
    iter_update_fn: base.Schedule
) -> base.GradientTransformation:
    def init_fn(params):
        del params
        return ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        del params
        updates = iter_update_fn(state.count, updates)
        return updates, ScaleByScheduleState(
            count=numerics.safe_int32_increment(state.count))
    return base.GradientTransformation(init_fn, update_fn)



@struct.dataclass
class CombineConfig:
    optimizer_a: OptimizerConfig
    optimizer_b: OptimizerConfig
    switch_percent: float

    def make_optimizer(self, iterations):
        switch_iteration = int(iterations*self.switch_percent)
        combined = combine(
            self.optimizer_a.make_optimizer(iterations),
            self.optimizer_b.make_optimizer(iterations)
        )
        switch = update_by_schedule(
            lambda i, updates: jax.lax.cond(i < switch_iteration, 
            lambda: updates[0], lambda: updates[1])
        )
        opt = optax.chain(combined, switch)
        return opt

@struct.dataclass
class SAMConfig:
    forward: OptimizerConfig = OptimizerConfig.default()
    backward: OptimizerConfig = SGDConfig(lr=5e-2) # rho = 0.05
    start_percent: float = 0.
    run_percent: float = 1.
    disable_backward: bool = False
    normalize: bool = True

    def make_optimizer(self, iterations):
        import optax.contrib as sam

        forward_opt = self.forward.make_optimizer(iterations)
        if self.disable_backward:
            return forward_opt
        backward_opt = self.backward.make_optimizer(iterations)
        # normalize before the backward optimizer
        backward_opt = optax.chain(sam.normalize(), backward_opt) if self.normalize else backward_opt
        if self.start_percent > 0 or self.run_percent < 1.:
            start_iter = int(self.start_percent * iterations)
            end_iter = int((iterations - start_iter)*self.run_percent) + start_iter
            backward_opt = optax.chain(
                backward_opt,
                optax.scale_by_schedule(
                    lambda i: jax.lax.cond(
                        jnp.logical_or(i < start_iter, i > end_iter), 
                    lambda: 0, lambda: 1))
            )
        return sam.sam(
            optimizer=forward_opt,
            adv_optimizer=backward_opt,
            reset_state=False,
            opaque_mode=True
        )


@struct.dataclass
class TrainConfig:
    batch_size: int = 32
    """The batch size to use for training."""
    epochs: int | None = None
    """The number of epochs to train for."""
    iterations: int | None = None
    """The number of iterations to train for."""
    optimizer: OptimizerConfig = None
    
    def fit(self, **kwargs):
        from stanza.train import fit
        data = kwargs.pop("data")
        if self.epochs is None and self.iterations is None:
            raise ValueError("Either epochs or iterations must be specified")
        iterations = self.iterations or (self.epochs * (len(data) // self.batch_size))
        return fit(
            data=data,
            batch_size=self.batch_size,
            max_iterations=iterations,
            optimizer=self.optimizer.make_optimizer(iterations),
            **kwargs
        )
    
    def parse(self, config: ConfigProvider) -> "TrainConfig":
        defaults = TrainConfig()
        res = config.get_struct(defaults, {"optimizer"})
        optimizer = config.get_cases("optimizer", "The optimizer to use", {
            "sgd": SGDConfig(),
            "adam": AdamConfig()
        }, "adam")
        return struct.replace(res, optimizer=optimizer)
