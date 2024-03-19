import jax.numpy as jnp
from jaxtyping import Array, Float

from lib.typing import ScalarFloat


def euler_to_rotation(angles: Float[Array, "3"]) -> Float[Array, "3 3"]:
    phi, theta, psi = angles
    first_row = jnp.array(
        [
            jnp.cos(psi) * jnp.cos(theta)
            - jnp.sin(phi) * jnp.sin(psi) * jnp.sin(theta),
            -jnp.cos(phi) * jnp.sin(psi),
            jnp.cos(psi) * jnp.sin(theta)
            + jnp.cos(theta) * jnp.sin(phi) * jnp.sin(psi),
        ]
    )
    second_row = jnp.array(
        [
            jnp.cos(theta) * jnp.sin(psi)
            + jnp.cos(psi) * jnp.sin(phi) * jnp.sin(theta),
            jnp.cos(phi) * jnp.cos(psi),
            jnp.sin(psi) * jnp.sin(theta)
            - jnp.cos(psi) * jnp.cos(theta) * jnp.sin(phi),
        ]
    )
    third_row = jnp.array(
        [-jnp.cos(phi) * jnp.sin(theta), jnp.sin(phi), jnp.cos(phi) * jnp.cos(theta)]
    )
    return jnp.stack([first_row, second_row, third_row])


def move_frame(angles: Float[Array, "3"]) -> Float[Array, "3 3"]:
    phi, theta, psi = angles
    first_row = jnp.array([jnp.cos(theta), 0, -jnp.cos(phi) * jnp.sin(theta)])
    second_row = jnp.array([0, 1, jnp.sin(phi)])
    third_row = jnp.array([jnp.sin(theta), 0, jnp.cos(phi) * jnp.cos(theta)])
    return jnp.stack([first_row, second_row, third_row])


def quadratic_cost(
    x: Float[Array, "12"],
    u: Float[Array, "4"],
    x_target: Float[Array, "12"],
    u_target: Float[Array, "4"],
    q: Float[Array, "12 12"],
    r: Float[Array, "4 4"],
) -> ScalarFloat:
    assert (
        x.ndim == u.ndim == 1
        and q.ndim == r.ndim == 2
        and x.shape[0] == q.shape[0]
        and u.shape[0] == r.shape[0]
    )
    norm_x = x - x_target
    norm_u = u - u_target
    return norm_x @ q @ norm_x + norm_u @ r @ norm_u
