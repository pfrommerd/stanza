import jax
import stanza
import inspect
import os

from functools import partial

import jax.experimental.host_callback
import rich.console
from rich.markup import escape
from typing import List, Any, Optional

# Topic management

# ---------------- Logging ---------------

TRACE = 'TRACE'
DEBUG = 'DEBUG'
INFO =  'INFO'
WARN =  'WARN'
ERROR = 'ERROR'

LEVEL_COLORS = {
    TRACE: 'green',
    DEBUG: 'cyan',
    INFO: 'white',
    WARN: 'yellow',
    ERROR: 'red'
}

console = rich.get_console()

JAX_PLACEHOLDER = object()

def _extract_stack_pos(offset=0):
    # Ignore the frame of this local helper
    offset += 1
    frame = inspect.currentframe()
    if frame is not None:
        # Use the faster currentframe where implemented
        while offset and frame is not None:
            frame = frame.f_back
            offset -= 1
        assert frame is not None
        return frame.f_code.co_filename, frame.f_lineno
    else:
        # Fallback to the slower stack
        frame_info = inspect.stack()[offset]
        return frame_info.filename, frame_info.lineno

def _log(*objects: Any,
        sep: str = " ",
        end: str = "\n",
        markup: Optional[bool] = None, highlight: Optional[bool] = None,
        filename: str = None,
        line_no: int = None
    ):
        if not objects:
            objects = (rich.console.NewLine(),)
        render_hooks = console._render_hooks[:]
        with console:
            renderables = console._collect_renderables(
                objects,
                sep,
                end,
                markup=markup,
                highlight=highlight,
            )
            link_path = None if filename.startswith("<") else os.path.abspath(filename)
            path = filename.rpartition(os.sep)[-1]
            renderables = [
                console._log_render(
                    console,
                    renderables,
                    log_time=console.get_datetime(),
                    path=path,
                    line_no=line_no,
                    link_path=link_path,
                )
            ]
            for hook in render_hooks:
                renderables = hook.process_renderables(renderables)
            new_segments: List[rich.console.Segment] = []
            extend = new_segments.extend
            render = console.render
            render_options = console.options
            for renderable in renderables:
                extend(render(renderable, render_options))
            buffer_extend = console._buffer.extend
            for line in rich.console.Segment.split_and_crop_lines(
                    new_segments, console.width, pad=False):
                buffer_extend(line)

# A jax-compatible logger
# This will bypass logging at compile time
# if the selected "topic" have not been enabled
class JaxLogger:
    # Host-side logging
    def _log_callback(self, level, msg, reg_comp, jax_comp,
                        tracing=False, highlight=True, filename=None, line_no=None):
        reg_args, reg_kwargs = reg_comp
        jax_args, jax_kwargs = jax_comp

        # reassemble args, kwargs
        args = []
        jax_iter = iter(jax_args)
        for a in reg_args:
            if a is JAX_PLACEHOLDER:
                args.append(next(jax_iter))
            else:
                args.append(a)
        kwargs = dict(reg_kwargs)
        kwargs.update(jax_kwargs)

        msg = msg.format(*args, **kwargs)
        level_color = LEVEL_COLORS.get(level, 'white')
        if tracing:
            msg = '[yellow]<Tracing>[/yellow] ' + msg

        # a version of console.log() which handles the stack frame correctly
        _log(f'[{level_color}]{level:6}[/{level_color}] - {msg}', 
            highlight=highlight, filename=filename, line_no=line_no)

    def log(self, level, msg, *args, highlight=True, show_tracing=False, only_tracing=False,
                _stack_offset=1, **kwargs):
        # split the arguments and kwargs
        # based on whether they are jax-compatible types or not
        reg_args = []
        jax_args = []
        for a in args:
            if stanza.is_jaxtype(type(a)):
                jax_args.append(a)
                reg_args.append(JAX_PLACEHOLDER)
            else:
                reg_args.append(a)
        reg_kwargs = {}
        jax_kwargs = {}
        for (k,v) in kwargs.items():
            if stanza.is_jaxtype(type(v)):
                jax_kwargs[k] = v
            else:
                reg_kwargs[k] = v
        tracing = isinstance(jax.numpy.array(0), jax.core.Tracer)
        filename, line_no = _extract_stack_pos(_stack_offset)
        if tracing and (show_tracing or only_tracing):
            self._log_callback(level, msg, (reg_args, reg_kwargs), (jax_args, jax_kwargs), 
                            tracing=True, filename=filename, line_no=line_no)
        if only_tracing:
            return
        if not tracing:
            self._log_callback(level, msg, (reg_args, reg_kwargs), (jax_args, jax_kwargs), 
                                filename=filename, line_no=line_no)
        else:
            jax.debug.callback(partial(self._log_callback, level, msg,
                                    (reg_args, reg_kwargs),
                                    highlight=highlight, filename=filename, line_no=line_no),  
                                    (args, kwargs), ordered=True)

    def trace(self, *args, **kwargs):
        return self.log(TRACE, *args, **kwargs, _stack_offset=2)

    def debug(self, *args, **kwargs):
        return self.log(DEBUG, *args, **kwargs, _stack_offset=2)

    def info(self, *args, **kwargs):
        return self.log(INFO, *args, **kwargs, _stack_offset=2)

    def warn(self, *args, **kwargs):
        return self.log(WARN, *args, **kwargs, _stack_offset=2)

    def error(self, *args, **kwargs):
        return self.log(ERROR, *args, **kwargs, _stack_offset=2)

logger = JaxLogger()

def _flat_items(d, prefix=''):
    for (k,v) in d.items():
        if isinstance(v, dict):
            yield from _flat_items(v, prefix=f'{prefix}{k}.')
        else:
            yield (f'{prefix}{k}',v)

from stanza.dataclasses import dataclass
from stanza.util.loop import Hook
from typing import Callable

import jax.numpy as jnp

@dataclass
class LoggerHook(Hook):
    condition_fn: Any
    stat_fn: Callable = lambda state: state.last_stats

    def init(self, state):
        return 0, state

    def run(self, hs, state):
        def log():
            stats = self.stat_fn(state)
            flat_stats = dict(_flat_items(stats))
            s = [f"{k}: {{}}" for k in flat_stats.keys()]
            fmt = "\n".join(s)
            logger.info("Iteration {}:\n" + fmt, state.iteration, *flat_stats.values())
        jax.lax.cond(jnp.logical_and(
            self.condition_fn(state),
            state.iteration != hs
        ), log, lambda: None)
        return state.iteration, state
    
    def finalize(self, hook_state, state):
        return hook_state, state