from functools import partial
from typing import NamedTuple
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax import jit
from jax.lax import cond

from examples.robotics.sim import OptimalCost, Simulator
from examples.robotics.quadrotor.utils import (
    euler_to_rotation,
    move_frame,
    quadratic_cost,
)
from lib.typing import ScalarFloat

ACTION_TARGET = jnp.array([1.766, 0.0, 0.0, 0.0])
"""Action which results in hovering (approximately) when in state 0 with undisturbed dynamics."""


class ModelFeedBackParams(NamedTuple):
    proportional_params: Float[Array, "4"] = jnp.zeros(4)
    derivative_params: Float[Array, "4"] = jnp.zeros(4)


class QuadrotorEuler:
    """
    Dynamics of quadrotor with 12 dimensional state space and 4 dimensional control
    Code adapted from : https://github.com/Bharath2/Quadrotor-Simulation
    Theory from: https://repository.upenn.edu/cgi/viewcontent.cgi?article=1705&context=edissertations
    Why 4 dimensional control: https://www.youtube.com/watch?v=UC8W3SfKGmg (talks about that at around 8min in video)
    Short description for 4 dimensional control:
          [ F  ]         [ F1 ]
          | M1 |  = A *  | F2 |
          | M2 |         | F3 |
          [ M3 ]         [ F4 ]
    """

    x_dim = 12
    u_dim = 4

    time_scaling: Float[Array, "1"]
    state_scaling: Float[Array, "12 12"]
    state_scaling_inv: Float[Array, "12 12"]
    control_scaling: Float[Array, "4 4"]
    control_scaling_inv: Float[Array, "4 4"]
    mass: float
    g: float
    arm_length: float
    height: float
    I: Float[Array, "3 3"]
    invI: Float[Array, "3 3"]
    minF: float
    maxF: float
    km: float
    kf: float
    r: float
    L: float
    H: float
    A: Float[Array, "4 4"]
    invA: Float[Array, "4 4"]
    body_shape: Float[Array, "6 4"]
    B: Float[Array, "2 4"]
    internal_control_scaling_inv: Float[Array, "4 4"]
    state_target: Float[Array, "12"]
    action_target: Float[Array, "4"]
    running_q: Float[Array, "12 12"]
    running_r: Float[Array, "4 4"]
    terminal_q: Float[Array, "12 12"]
    terminal_r: Float[Array, "4 4"]

    def __init__(
        self,
        state_target: Float[Array, "12"],
        action_target: Float[Array, "4"],
        time_scaling: Float[Array, "1"] | None = None,
        state_scaling: Float[Array, "12 12"] | None = None,
        control_scaling: Float[Array, "4 4"] | None = None,
    ):
        self.x_dim = 12
        self.u_dim = 4
        if time_scaling is None:
            time_scaling = jnp.ones(shape=(1,))
        if state_scaling is None:
            state_scaling = jnp.eye(self.x_dim)
        if control_scaling is None:
            control_scaling = jnp.eye(self.u_dim)
        self.time_scaling = time_scaling
        self.state_scaling = state_scaling
        self.state_scaling_inv = jnp.linalg.inv(state_scaling)
        self.control_scaling = control_scaling
        self.control_scaling_inv = jnp.linalg.inv(control_scaling)

        self.mass = 0.18  # kg
        self.g = 9.81  # m/s^2
        self.arm_length = 0.086  # meter
        self.height = 0.05

        self.I = jnp.array(
            [(0.00025, 0, 2.55e-6), (0, 0.000232, 0), (2.55e-6, 0, 0.0003738)]
        )

        self.invI = jnp.linalg.inv(self.I)

        self.minF = 0.0
        self.maxF = 2.0 * self.mass * self.g

        self.km = 1.5e-9
        self.kf = 6.11e-8
        self.r = self.km / self.kf

        self.L = self.arm_length
        self.H = self.height
        #  [ F  ]         [ F1 ]
        #  | M1 |  = A *  | F2 |
        #  | M2 |         | F3 |
        #  [ M3 ]         [ F4 ]
        self.A = jnp.array(
            [
                [1, 1, 1, 1],
                [0, self.L, 0, -self.L],
                [-self.L, 0, self.L, 0],
                [self.r, -self.r, self.r, -self.r],
            ]
        )

        self.invA = jnp.linalg.inv(self.A)

        self.body_frame = jnp.array(
            [
                (self.L, 0, 0, 1),
                (0, self.L, 0, 1),
                (-self.L, 0, 0, 1),
                (0, -self.L, 0, 1),
                (0, 0, 0, 1),
                (0, 0, self.H, 1),
            ]
        )

        self.B = jnp.array([[0, self.L, 0, -self.L], [-self.L, 0, self.L, 0]])

        self.internal_control_scaling_inv = jnp.diag(
            jnp.array([1, 2 * 1e-4, 2 * 1e-4, 1e-3], dtype=jnp.float64)
        )

        # Cost parameters:
        self.state_target = state_target
        self.action_target = action_target
        self.running_q = 1.0 * jnp.diag(
            jnp.array([1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 1, 0.1, 0.1], dtype=jnp.float64)
        )
        self.running_r = (
            1e-2 * 1.0 * jnp.diag(jnp.array([5.0, 0.8, 0.8, 0.3], dtype=jnp.float64))
        )
        self.terminal_q = 5.0 * jnp.eye(self.x_dim)
        self.terminal_r = 0.0 * jnp.eye(self.u_dim)

    @partial(jit, static_argnums=0)
    def _ode(
        self, state: Float[Array, "12"], u: Float[Array, "4"]
    ) -> Float[Array, "12"]:
        # u = self.scaling_u_inv @ u
        # Here we have to decide in which coordinate system we will operate
        # If we operate with F1, F2, F3 and F4 we need to run
        # u = self.A @ u
        u = self.internal_control_scaling_inv @ u
        F, M = u[0], u[1:]
        x, y, z, xdot, ydot, zdot, phi, theta, psi, p, q, r = state
        angles = jnp.array([phi, theta, psi])
        wRb = euler_to_rotation(angles)
        # acceleration - Newton's second law of motion
        accel = (
            1.0
            / self.mass
            * (
                wRb.dot(jnp.array([[0, 0, F]]).T)
                - jnp.array([[0, 0, self.mass * self.g]]).T
            )
        )
        # angular acceleration - Euler's equation of motion
        # https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
        omega = jnp.array([p, q, r])
        angles_dot = jnp.linalg.inv(move_frame(angles)) @ omega
        pqrdot = self.invI.dot(M.flatten() - jnp.cross(omega, self.I.dot(omega)))
        state_dot_0 = xdot
        state_dot_1 = ydot
        state_dot_2 = zdot
        state_dot_3 = accel[0].reshape()
        state_dot_4 = accel[1].reshape()
        state_dot_5 = accel[2].reshape()
        state_dot_6 = angles_dot[0]
        state_dot_7 = angles_dot[1]
        state_dot_8 = angles_dot[2]
        state_dot_9 = pqrdot[0]
        state_dot_10 = pqrdot[1]
        state_dot_11 = pqrdot[2]
        return jnp.array(
            [
                state_dot_0,
                state_dot_1,
                state_dot_2,
                state_dot_3,
                state_dot_4,
                state_dot_5,
                state_dot_6,
                state_dot_7,
                state_dot_8,
                state_dot_9,
                state_dot_10,
                state_dot_11,
            ]
        )

    @partial(jit, static_argnums=0)
    def ode(
        self,
        x: Float[Array, "12"],
        u: Float[Array, "4"],
        disturbance_params: ModelFeedBackParams = ModelFeedBackParams(),
        feedback_params: ModelFeedBackParams = ModelFeedBackParams(),
    ) -> Float[Array, "12"]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        x = self.state_scaling_inv @ x
        u = self.control_scaling_inv @ u
        position_diff = jnp.concatenate(
            [(x[2] - self.state_target[2]).reshape(1), x[6:9] - self.state_target[6:9]]
        )
        velocity_diff = jnp.concatenate(
            [
                (x[5] - self.state_target[5]).reshape(1),
                x[9:12] - self.state_target[9:12],
            ]
        )
        disturbance = (
            disturbance_params.proportional_params - feedback_params.proportional_params
        ) * position_diff + (
            disturbance_params.derivative_params - feedback_params.derivative_params
        ) * velocity_diff
        return (
            self.state_scaling
            @ self._ode(x, u + disturbance)
            / self.time_scaling.reshape()
        )

    @partial(jit, static_argnums=0)
    def running_cost(self, x: Float[Array, "12"], u: Float[Array, "4"]) -> ScalarFloat:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        x = self.state_scaling_inv @ x
        u = self.control_scaling_inv @ u
        return self._running_cost(x, u) / self.time_scaling.reshape()

    @partial(jit, static_argnums=0)
    def terminal_cost(self, x: Float[Array, "12"], u: Float[Array, "4"]) -> ScalarFloat:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        x = self.state_scaling_inv @ x
        u = self.control_scaling_inv @ u
        return self._terminal_cost(x, u)

    @partial(jit, static_argnums=0)
    def _running_cost(self, x: Float[Array, "12"], u: Float[Array, "4"]) -> ScalarFloat:
        return quadratic_cost(
            x,
            u,
            x_target=self.state_target,
            u_target=self.action_target,
            q=self.running_q,
            r=self.running_r,
        )

    @partial(jit, static_argnums=0)
    def _terminal_cost(
        self, x: Float[Array, "12"], u: Float[Array, "4"]
    ) -> ScalarFloat:
        return quadratic_cost(
            x,
            u,
            x_target=self.state_target,
            u_target=self.action_target,
            q=self.terminal_q,
            r=self.terminal_r,
        )


class QuadrotorOptimalCost(OptimalCost):
    x_dim = 12
    u_dim = 4
    dynamics: QuadrotorEuler

    def __init__(self, sim_params: dict = {}, **kwargs):
        self.dynamics = QuadrotorEuler(**sim_params)
        super().__init__(**kwargs)

    @partial(jit, static_argnums=0)
    def running_cost(
        self, x: Float[Array, "12"], u: Float[Array, "4"], t: ScalarFloat
    ) -> ScalarFloat:
        return self.dt * self.dynamics.running_cost(x, u)

    @partial(jit, static_argnums=0)
    def terminal_cost(
        self, x: Float[Array, "12"], u: Float[Array, "4"], t: ScalarFloat
    ) -> ScalarFloat:
        return self.dynamics.terminal_cost(x, u)

    @partial(jit, static_argnums=0)
    def cost(
        self,
        x: Float[Array, "12"],
        u: Float[Array, "4"],
        t: ScalarFloat,
        params: dict | None = None,
    ):
        return cond(
            t == self.num_nodes,
            self.terminal_cost,
            self.running_cost,
            x,
            u,
            t,
        )

    @partial(jit, static_argnums=0)
    def next_step(
        self,
        x: Float[Array, "12"],
        u: Float[Array, "4"],
        t: ScalarFloat,
        params: dict | None = None,
    ) -> Float[Array, "12"]:
        dynamics_params = {}
        if params is not None:
            for key in ["disturbance_params", "feedback_params"]:
                if key in params.keys():
                    dynamics_params[key] = params[key]
        return x + self.dt * self.dynamics.ode(x, u, **dynamics_params)

    @classmethod
    @partial(jit, static_argnums=0)
    def _constraint(cls, x: Float[Array, "x_dim"]) -> ScalarFloat:
        return jnp.maximum(0, jnp.nan_to_num(x[2], nan=0.0))


if __name__ == "__main__":
    from jax._src.config import config

    config.update("jax_enable_x64", True)

    x0 = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_target = jnp.array(
        [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    Simulator.plot_feedback_params_safety(
        OC=QuadrotorOptimalCost,
        name="quadrotor",
        x0=x0,
        state_target=state_target,
        action_target=ACTION_TARGET,
        fmin=0.5,
        disturbance_scale=1.0,
        ylabel="Minimum z-position",
    )
