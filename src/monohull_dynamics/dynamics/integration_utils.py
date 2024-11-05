from functools import partial

import numpy as np
import jax.numpy as jnp
import jax


def gauss_legendre_second_order_jax_vector(f, x0, v0, h, tol=1e-10, max_iter=100):
    """
    Second-order Gauss-Legendre integrator (implicit midpoint method) using Newton's method.

    Parameters:
    f : callable
        Function representing dv/dt = f(x, v).
    x0, v0 : array-like
        Initial position and velocity.
    h : float
        Time step size.
    tol : float, optional
        Tolerance for the Newton solver.
    max_iter : int, optional
        Maximum number of iterations for the solver.

    Returns:
    x_next, v_next : array-like
        Updated position and velocity after time step h.
    """

    # Define the function G(v_est) for Newton's method
    def G(v_est):
        x_est = x0 + (h / 2) * v_est
        return v_est - v0 - (h / 2) * f(x_est, v_est)

    # Initial guess for v_est (can be v0)
    v_est_init = v0

    def cond_fun(carry):
        v_est, v_est_prev, iters, _ = carry
        # Check for convergence based on the norm of the difference
        converged = jnp.linalg.norm(v_est - v_est_prev) < tol
        # jax.debug.print("{c}", c=converged)
        return (iters < max_iter) & (~converged)

    def body_fun(carry):
        v_est_prev, _, iters, _ = carry

        # Compute G(v_est_prev) and its Jacobian
        G_value = G(v_est_prev)
        J = jax.jacfwd(G)(v_est_prev)

        # Solve for delta_v: J * delta_v = -G_value
        delta_v = -jnp.linalg.solve(J, G_value)

        v_est = v_est_prev + delta_v

        return v_est, v_est_prev, iters + 1, None

    # Run the Newton iteration loop
    v_est, _, _, _ = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (v_est_init, jnp.inf * v_est_init, 0, None)
    )

    # Compute the updated position and velocity
    x_est = x0 + (h / 2) * v_est
    x_next = x0 + h * v_est
    v_next = v0 + h * f(x_est, v_est)

    return x_next, v_next


@partial(jax.jit, static_argnums=(0,))
def gauss_legendre_fourth_order_jax_vector(f, x0, v0, h, tol=1e-12, max_iter=10):
    """
    Fourth-order Gauss-Legendre integrator using Newton's method.

    Parameters:
    f : callable
        Function representing dv/dt = f(x, v).
    x0, v0 : array-like
        Initial position and velocity.
    h : float
        Time step size.
    tol : float, optional
        Tolerance for the Newton solver.
    max_iter : int, optional
        Maximum number of iterations for the solver.

    Returns:
    x_next, v_next : array-like
        Updated position and velocity after time step h.
    """
    sqrt3 = jnp.sqrt(3.0)
    a11 = 0.25
    a12 = 0.25 - sqrt3 / 6.0
    a21 = 0.25 + sqrt3 / 6.0
    a22 = 0.25
    b1 = b2 = 0.5

    # Initial guesses for v1 and v2
    v1_est = v0
    v2_est = v0

    # Combine v1 and v2 into a single vector for Newton's method
    v_est_init = jnp.concatenate([v1_est, v2_est])

    # Define the function G(v_est) for Newton's method
    def G(v_est):
        v1_est, v2_est = jnp.split(v_est, 2)
        x1_est = x0 + h * (a11 * v1_est + a12 * v2_est)
        x2_est = x0 + h * (a21 * v1_est + a22 * v2_est)

        f1 = f(x1_est, v1_est)
        f2 = f(x2_est, v2_est)

        G1 = v1_est - v0 - h * (a11 * f1 + a12 * f2)
        G2 = v2_est - v0 - h * (a21 * f1 + a22 * f2)

        return jnp.concatenate([G1, G2])

    def cond_fun(carry):
        v_est, v_est_prev, iters = carry
        # Check for convergence based on the norm of the difference
        converged = jnp.linalg.norm(v_est - v_est_prev) < tol
        return (iters < max_iter) & (~converged)

    def body_fun(carry):
        v_est_prev, _, iters = carry

        # Compute G(v_est_prev) and its Jacobian
        G_value = G(v_est_prev)
        J = jax.jacfwd(G)(v_est_prev)

        # Solve for delta_v: J * delta_v = -G_value
        delta_v = -jnp.linalg.solve(J, G_value)
        delta_v = jnp.nan_to_num(delta_v, nan=h, posinf=h, neginf=h)

        v_est = v_est_prev + delta_v

        return v_est, v_est_prev, iters + 1

    # # Run the Newton iteration loop
    v_est, _, _ = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (v_est_init, v_est_init * jnp.inf, 0)
    )

    # Extract v1_est and v2_est from v_est
    v1_est, v2_est = jnp.split(v_est, 2)

    # Compute x_next and v_next
    x_next = x0 + h * (b1 * v1_est + b2 * v2_est)

    # Recompute x1_est and x2_est for final f1 and f2
    x1_est = x0 + h * (a11 * v1_est + a12 * v2_est)
    x2_est = x0 + h * (a21 * v1_est + a22 * v2_est)

    f1 = f(x1_est, v1_est)
    f2 = f(x2_est, v2_est)

    v_next = v0 + h * (b1 * f1 + b2 * f2)

    return x_next, v_next

# Example usage:
if __name__ == "__main__":
    # Example ODE: dv/dt = -x (simple harmonic oscillator)
    def f(x, v):
        return -x


    x0, v0 = 1.0, 0.0  # Initial conditions
    h = 0.1  # Time step

    xs = [jnp.array([x0,x0])]
    vs = [jnp.array([v0,v0])]
    _as = [f(xs[0], vs[0])]

    for i in range(100):
        x_, v_ = gauss_legendre_fourth_order_jax_vector(f, xs[-1], vs[-1], h)
        xs.append(x_)
        vs.append(v_)
        a = f(xs[-1], vs[-1])
        _as.append(a)

    print(_as[0], _as[1])

    from matplotlib import pyplot as plt
    xs = np.array(xs)
    vs = np.array(vs)
    _as = np.array(_as)
    print(xs.shape)
    plt.plot(range(101), xs[:, 0], label="x")
    plt.plot(range(101), vs[:, 0], label="v")
    plt.plot(range(101), _as[:, 0], label="a")
    plt.legend()
    plt.show()
