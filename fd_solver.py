import numpy as np


def compute_cfl_dt(dx, c_max, cfl=0.5):
    """
    Compute a stable time step for the 2D explicit
    acoustic finite-difference scheme.

    Stability condition:

        c * dt / dx <= CFL / sqrt(2)

    Parameters
    ----------
    dx : float
        Spatial grid spacing in km.

    c_max : float
        Maximum velocity in km/s.

    cfl : float
        Chosen CFL safety factor.

    Returns
    -------
    float
        Stable time step in seconds.
    """

    return cfl * dx / c_max


def run_fd_solver(
    velocity,
    source_function,
    x_min=0.0,
    x_max=2.0,
    y_min=-1.0,
    y_max=2.0,
    nx=301,
    ny=451,
    t_max=1.0,
    cfl=0.5,
):
    """
    Solve the 2D acoustic wave equation using an explicit
    second-order finite-difference scheme.

    PDE:

        u_tt = c^2 (u_xx + u_yy) + q

    Parameters
    ----------
    velocity : callable
        Function velocity(x, y) returning velocity in km/s.

    source_function : callable
        Function source(x, y, t).

    x_min, x_max : float
        x-domain in km.

    y_min, y_max : float
        y-domain in km.

    nx, ny : int
        Number of grid points in x and y.

    t_max : float
        Final simulation time in seconds.

    cfl : float
        CFL safety factor.

    Returns
    -------
    dict
        Simulation results.
    """

    # -----------------------------------------------------
    # Spatial grid
    # -----------------------------------------------------

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    if not np.isclose(dx, dy):
        raise ValueError(
            f"This solver requires dx = dy. "
            f"Got dx={dx}, dy={dy}."
        )

    # -----------------------------------------------------
    # 2D coordinates
    # -----------------------------------------------------

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij",
    )

    # -----------------------------------------------------
    # Velocity model
    # -----------------------------------------------------

    c = velocity(X, Y)

    c_max = np.max(c)

    # -----------------------------------------------------
    # Stable time step
    # -----------------------------------------------------

    dt = compute_cfl_dt(
        dx=dx,
        c_max=c_max,
        cfl=cfl,
    )

    nt = int(np.ceil(t_max / dt)) + 1

    # Adjust dt so that t = t_max exactly
    dt = t_max / (nt - 1)

    # Actual CFL after adjustment
    actual_cfl = c_max * dt / dx

    if actual_cfl > 1.0 / np.sqrt(2.0):
        raise ValueError(
            f"Unstable CFL condition: {actual_cfl:.6f}"
        )

    t = np.linspace(
        0.0,
        t_max,
        nt,
    )

    # -----------------------------------------------------
    # Wavefield storage
    # -----------------------------------------------------

    wavefield = np.zeros(
        (nt, nx, ny),
        dtype=np.float32,
    )

    # Previous and current wavefields
    u_prev = np.zeros(
        (nx, ny),
        dtype=np.float32,
    )

    u_curr = np.zeros(
        (nx, ny),
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # FD coefficient
    # -----------------------------------------------------

    coefficient = (
        (c.astype(np.float32) * dt / dx) ** 2
    )

    # -----------------------------------------------------
    # Time stepping
    # -----------------------------------------------------

    for n in range(nt - 1):

        u_next = np.zeros_like(u_curr)

        # -------------------------------------------------
        # Interior points
        # -------------------------------------------------

        laplacian = (
            u_curr[2:, 1:-1]
            + u_curr[:-2, 1:-1]
            + u_curr[1:-1, 2:]
            + u_curr[1:-1, :-2]
            - 4.0 * u_curr[1:-1, 1:-1]
        )

        source = source_function(
            X[1:-1, 1:-1],
            Y[1:-1, 1:-1],
            t[n],
        )

        u_next[1:-1, 1:-1] = (
            2.0 * u_curr[1:-1, 1:-1]
            - u_prev[1:-1, 1:-1]
            + coefficient[1:-1, 1:-1] * laplacian
            + dt**2 * source
        )

        # -------------------------------------------------
        # First-order absorbing boundaries
        #
        # u_t + c * du/dn = 0
        # -------------------------------------------------

        # Left boundary: x = x_min
        # n = (-1, 0)
        u_next[0, 1:-1] = (
            u_curr[0, 1:-1]
            + dt
            * c[0, 1:-1]
            * (
                u_curr[1, 1:-1]
                - u_curr[0, 1:-1]
            )
            / dx
        )

        # Right boundary: x = x_max
        # n = (+1, 0)
        u_next[-1, 1:-1] = (
            u_curr[-1, 1:-1]
            - dt
            * c[-1, 1:-1]
            * (
                u_curr[-1, 1:-1]
                - u_curr[-2, 1:-1]
            )
            / dx
        )

        # Bottom boundary: y = y_min
        # n = (0, -1)
        u_next[1:-1, 0] = (
            u_curr[1:-1, 0]
            + dt
            * c[1:-1, 0]
            * (
                u_curr[1:-1, 1]
                - u_curr[1:-1, 0]
            )
            / dy
        )

        # Top boundary: y = y_max
        # n = (0, +1)
        u_next[1:-1, -1] = (
            u_curr[1:-1, -1]
            - dt
            * c[1:-1, -1]
            * (
                u_curr[1:-1, -1]
                - u_curr[1:-1, -2]
            )
            / dy
        )

        # -------------------------------------------------
        # Corners
        # -------------------------------------------------

        u_next[0, 0] = (
            u_next[0, 1]
            + u_next[1, 0]
        ) / 2.0

        u_next[0, -1] = (
            u_next[0, -2]
            + u_next[1, -1]
        ) / 2.0

        u_next[-1, 0] = (
            u_next[-1, 1]
            + u_next[-2, 0]
        ) / 2.0

        u_next[-1, -1] = (
            u_next[-1, -2]
            + u_next[-2, -1]
        ) / 2.0

        # -------------------------------------------------
        # Store and advance
        # -------------------------------------------------

        wavefield[n + 1] = u_next

        u_prev = u_curr
        u_curr = u_next

    return {
        "x": x,
        "y": y,
        "t": t,
        "wavefield": wavefield,
        "velocity": c,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "cfl": actual_cfl,
    }