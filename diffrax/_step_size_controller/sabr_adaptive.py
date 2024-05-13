from typing import Callable, Optional, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree, Real

from .base import AbstractStepSizeController
from .._custom_types import Args, BoolScalarLike, IntScalarLike, RealScalarLike, VF, Y
from .._misc import upcast_or_raise
from .._solution import RESULTS
from .._term import AbstractTerm

_ControllerState: TypeAlias = None


def _none_or_array(x):
    if x is None:
        return None
    else:
        return jnp.asarray(x)


@jax.jit
def inv_expx_min1_by_x(z):
    initial = jnp.log(z)

    def step(_, x):
        exp_x = jnp.exp(x)
        x_exp_x = x * exp_x
        return x * (1 - (exp_x - 1 - x * z) / (x_exp_x - exp_x + 1))

    return jax.lax.fori_loop(0, 10, step, initial)


class SABRController(AbstractStepSizeController[None, Optional[RealScalarLike]]):
    """Step size controller for the CIR process."""

    ctol: RealScalarLike
    dtmax: RealScalarLike
    dtmin: RealScalarLike
    step_ts: Optional[Real[Array, " steps"]] = eqx.field(
        default=None, converter=_none_or_array
    )
    previsible: bool = eqx.field(default=False)
    euler: bool = eqx.field(default=False)

    def wrap(self, direction: IntScalarLike) -> "AbstractStepSizeController":
        return self

    def desired_step_size(self, v):
        z = 1 + self.ctol * jnp.exp(-2 * v)
        if self.euler:
            step_size = inv_expx_min1_by_x(z)
        else:
            step_size = jnp.log(z)

        step_size = jnp.nan_to_num(
            step_size, nan=self.dtmin, posinf=self.dtmax, neginf=self.dtmin
        )
        return jnp.clip(step_size, self.dtmin, self.dtmax)

    def _clip_step_ts(self, t0: RealScalarLike, t1: RealScalarLike) -> RealScalarLike:
        # Copied from PIDController
        if self.step_ts is None:
            return t1
        step_ts0 = upcast_or_raise(
            self.step_ts,
            t0,
            "`SABRController.step_ts`",
            "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
        )
        step_ts1 = upcast_or_raise(
            self.step_ts,
            t1,
            "`SABRController.step_ts`",
            "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
        )
        t0_index = jnp.searchsorted(step_ts0, t0, side="right")
        t1_index = jnp.searchsorted(step_ts1, t1, side="right")
        t1 = jnp.where(
            t0_index < t1_index,
            step_ts1[jnp.minimum(t0_index, len(self.step_ts) - 1)],
            t1,
        )
        return t1

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        dt0: Optional[RealScalarLike],
        args: Args,
        func: Callable[[PyTree[AbstractTerm], RealScalarLike, Y, Args], VF],
        error_order: Optional[RealScalarLike],
    ) -> tuple[RealScalarLike, None]:
        del terms, t1, dt0, args, func, error_order
        assert y0.shape == (2,)
        step_size = self.desired_step_size(y0[1])
        t1 = self._clip_step_ts(t0, t0 + step_size)
        return t1, None

    def adapt_step_size(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        y1_candidate: Y,
        args: Args,
        y_error: Optional[Y],
        error_order: RealScalarLike,
        controller_state: _ControllerState,
    ) -> tuple[
        BoolScalarLike,
        RealScalarLike,
        RealScalarLike,
        BoolScalarLike,
        _ControllerState,
        RESULTS,
    ]:
        del args, y_error, error_order, controller_state
        assert y0.shape == (2,)
        v1 = y1_candidate[1]
        accepted_desired = self.desired_step_size(v1)
        if self.previsible:
            new_t1 = self._clip_step_ts(t1, t1 + accepted_desired)
            return True, t1, new_t1, False, None, RESULTS.successful

        v0 = y0[1]
        v_max = jnp.maximum(v0, v1)
        desired = self.desired_step_size(v_max)
        accept = t1 - t0 < 1.1 * desired
        new_t0 = jnp.where(accept, t1, t0)
        new_dt = jnp.where(accept, accepted_desired, desired)
        new_dt = jnp.clip(new_dt, self.dtmin, self.dtmax)
        new_t1 = self._clip_step_ts(new_t0, new_t0 + new_dt)
        return accept, new_t0, new_t1, False, None, RESULTS.successful
