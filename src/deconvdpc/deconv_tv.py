from typing import Optional
import numpy as np


def _gradient(M: np.ndarray, bound: str = "sym", order: int = 1) -> np.ndarray:
    """Compute image gradient using finite differences.

    Computes forward, backward, or centered difference gradient.
    Supports symmetric and periodic boundary conditions.

    Args:
        M: Input image (2D or 3D ndarray).
        bound: Boundary condition - "sym" (symmetric) or "per" (periodic).
            Default is "sym".
        order: Difference order - 1 (forward/backward) or 2 (centered).
            Default is 1.

    Returns:
        Gradient array with shape M.shape + (ndim,) where ndim is the
        number of spatial dimensions.
    """
    ndims = np.ndim(M)

    if bound == "sym":
        nx = M.shape[0]
        if order == 1:
            fx = M[np.hstack((np.arange(1, nx), [nx - 1])), :] - M
        else:
            fx = (
                M[np.hstack((np.arange(1, nx), [nx - 1])), :]
                - M[np.hstack(([0], np.arange(0, nx - 1))), :]
            ) / 2.0
            fx[0, :] = M[1, :] - M[0, :]
            fx[nx - 1, :] = M[nx - 1, :] - M[nx - 2, :]

        if ndims >= 2:
            ny = M.shape[1]
            if order == 1:
                fy = M[:, np.hstack((np.arange(1, ny), [ny - 1]))] - M
            else:
                fy = (
                    M[:, np.hstack((np.arange(1, ny), [ny - 1]))]
                    - M[:, np.hstack(([0], np.arange(ny - 1)))]
                ) / 2.0
                fy[:, 0] = M[:, 1] - M[:, 0]
                fy[:, ny - 1] = M[:, ny - 1] - M[:, ny - 2]

        if ndims >= 3:
            nz = M.shape[2]
            if order == 1:
                fz = M[:, :, np.hstack((np.arange(1, nz), [nz - 1]))] - M
            else:
                fz = (
                    M[:, :, np.hstack((np.arange(1, nz), [nz - 1]))]
                    - M[:, :, np.hstack(([0], np.arange(nz - 1)))]
                ) / 2.0
                fz[:, :, 0] = M[:, :, 1] - M[:, :, 0]
                fz[:, :, nz - 1] = M[:, :, nz - 1] - M[:, :, nz - 2]
    else:
        # Periodic boundary
        nx = M.shape[0]
        if order == 1:
            fx = M[np.hstack((np.arange(1, nx), [0])), :] - M
        else:
            fx = (
                M[np.hstack((np.arange(1, nx), [0])), :]
                - M[np.hstack(([nx - 1], np.arange(nx - 1))), :]
            ) / 2.0

        if ndims >= 2:
            ny = M.shape[1]
            if order == 1:
                fy = M[:, np.hstack((np.arange(1, ny), [0]))] - M
            else:
                fy = (
                    M[:, np.hstack((np.arange(1, ny), [0]))]
                    - M[:, np.hstack(([ny - 1], np.arange(ny - 1)))]
                ) / 2.0

        if ndims >= 3:
            nz = M.shape[2]
            if order == 1:
                fz = M[:, :, np.hstack((np.arange(1, nz), [0]))] - M
            else:
                fz = (
                    M[:, :, np.hstack((np.arange(1, nz), [0]))]
                    - M[:, :, np.hstack(([nz - 1], np.arange(nz - 1)))]
                ) / 2.0

    # Stack gradients
    if ndims == 2:
        result = np.stack((fx, fy), axis=2)
    elif ndims == 3:
        result = np.stack((fx, fy, fz), axis=3)
    else:
        result = fx

    return result


def _divergence(
    Px: np.ndarray, Py: Optional[np.ndarray] = None, bound: str = "sym", order: int = 1
) -> np.ndarray:
    """Compute divergence using finite differences.

    Adjoint operator to gradient. Satisfies <grad(f), g> = <f, -div(g)>.

    Args:
        Px: X-component of vector field, or full gradient array.
        Py: Y-component of vector field (optional).
        bound: Boundary condition - "sym" (symmetric) or "per" (periodic).
        order: Difference order - 1 or 2.

    Returns:
        Divergence array with same spatial dimensions as input.
    """
    Pz = None

    # Extract components if needed
    ndims = np.ndim(Px)
    if ndims >= 3:
        if ndims == 3:
            Py = Px[:, :, 1]
            Px = Px[:, :, 0]
            ndims = 2
        else:
            Pz = Px[:, :, :, 2]
            Py = Px[:, :, :, 1]
            Px = Px[:, :, :, 0]
            ndims = 3

    if bound == "sym":
        nx = Px.shape[0]
        if order == 1:
            fx = Px - Px[np.hstack(([0], np.arange(0, nx - 1))), :]
            fx[0, :] = Px[0, :]
            fx[nx - 1, :] = -Px[nx - 2, :]

            if ndims >= 2:
                ny = Py.shape[1]
                fy = Py - Py[:, np.hstack(([0], np.arange(0, ny - 1)))]
                fy[:, 0] = Py[:, 0]
                fy[:, ny - 1] = -Py[:, ny - 2]

            if ndims >= 3:
                nz = Pz.shape[2]
                fz = Pz - Pz[:, :, np.hstack(([0], np.arange(0, nz - 1)))]
                fz[:, :, 0] = Pz[:, :, 0]
                fz[:, :, nz - 1] = -Pz[:, :, nz - 2]
        else:
            # Centered differences with boundary handling
            nx = Px.shape[0]
            fx = (
                Px[np.hstack((np.arange(1, nx), [nx - 1])), :]
                - Px[np.hstack(([0], np.arange(0, nx - 1))), :]
            ) / 2.0
            fx[0, :] = Px[1, :] / 2.0 + Px[0, :]
            fx[1, :] = Px[2, :] / 2.0 - Px[0, :]
            fx[nx - 1, :] = -Px[nx - 1, :] - Px[nx - 2, :] / 2.0
            fx[nx - 2, :] = Px[nx - 1, :] - Px[nx - 3, :] / 2.0

            if ndims >= 2:
                ny = Py.shape[1]
                fy = (
                    Py[:, np.hstack((np.arange(1, ny), [ny - 1]))]
                    - Py[:, np.hstack(([0], np.arange(0, ny - 1)))]
                ) / 2.0
                fy[:, 0] = Py[:, 1] / 2.0 + Py[:, 0]
                fy[:, 1] = Py[:, 2] / 2.0 - Py[:, 0]
                fy[:, ny - 1] = -Py[:, ny - 1] - Py[:, ny - 2] / 2.0
                fy[:, ny - 2] = Py[:, ny - 1] - Py[:, ny - 3] / 2.0

            if ndims >= 3:
                nz = Pz.shape[2]
                fz = (
                    Pz[:, :, np.hstack((np.arange(1, nz), [nz - 1]))]
                    - Pz[:, :, np.hstack(([0], np.arange(0, nz - 1)))]
                ) / 2.0
                fz[:, :, 0] = Pz[:, :, 1] / 2.0 + Pz[:, :, 0]
                fz[:, :, 1] = Pz[:, :, 2] / 2.0 - Pz[:, :, 0]
                fz[:, :, nz - 1] = -Pz[:, :, nz - 1] - Pz[:, :, nz - 2] / 2.0
                fz[:, :, nz - 2] = Pz[:, :, nz - 1] - Pz[:, :, nz - 3] / 2.0
    else:
        # Periodic boundary conditions
        if order == 1:
            nx = Px.shape[0]
            fx = Px - Px[np.hstack(([nx - 1], np.arange(0, nx - 1))), :]

            if ndims >= 2:
                ny = Py.shape[1]
                fy = Py - Py[:, np.hstack(([ny - 1], np.arange(0, ny - 1)))]

            if ndims >= 3:
                nz = Pz.shape[2]
                fz = Pz - Pz[:, :, np.hstack(([nz - 1], np.arange(0, nz - 1)))]
        else:
            nx = Px.shape[0]
            fx = (
                Px[np.hstack((np.arange(1, nx), [0])), :]
                - Px[np.hstack(([nx - 1], np.arange(0, nx - 1))), :]
            )

            if ndims >= 2:
                ny = Py.shape[1]
                fy = (
                    Py[:, np.hstack((np.arange(1, ny), [0]))]
                    - Py[:, np.hstack(([ny - 1], np.arange(0, ny - 1)))]
                )

            if ndims >= 3:
                nz = Pz.shape[2]
                fz = (
                    Pz[:, :, np.hstack((np.arange(1, nz), [0]))]
                    - Pz[:, :, np.hstack(([nz - 1], np.arange(0, nz - 1)))]
                )

    # Combine components
    if ndims == 3:
        fd = fx + fy + fz
    elif ndims == 2:
        fd = fx + fy
    else:
        fd = fx

    return fd


def _tv_denoise(
    y: np.ndarray,
    epsilon: float,
    lambda_: float,
    iterations: int,
) -> np.ndarray:
    """Total variation denoising using iterative gradient descent.

    Solves the TV denoising problem:
        min_x { ||x - y||^2 + lambda * TV(x) }

    Args:
        y: Input image (2D ndarray).
        epsilon: Regularization parameter for smooth approximation of TV norm.
        lambda_: TV regularization strength.
        iterations: Number of iterations.

    Returns:
        Denoised image with same shape as input.
    """
    # Compute step size
    tau = 1.9 / (1 + lambda_ * 8 / epsilon)

    fTV = y.copy()

    for _ in range(iterations):
        # Compute gradient
        grad = _gradient(fTV)

        # Compute gradient norm
        d = np.sqrt(np.sum(grad**2, axis=2))
        deps = np.sqrt(epsilon**2 + d**2)

        # Compute divergence of normalized gradient
        grad_normalized = grad / np.expand_dims(deps, axis=-1)
        G0 = -_divergence(grad_normalized)

        # Gradient descent step
        G = fTV - y + lambda_ * G0
        fTV = fTV - tau * G

    return fTV


def tv_deconvolution(
    y: np.ndarray,
    regul_param: float,
    max_iter: int = 400,
    psf: np.ndarray = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]]),
) -> np.ndarray:
    """Total variation deconvolution using plug-and-play algorithm.

    Performs image deconvolution with total variation regularization using
    the alternating direction method of multipliers (ADMM-like approach).

    Args:
        y: Blurred/degraded image (2D ndarray).
        psf: Point spread function (PSF, 2D ndarray).
        regularization_param: TV regularization strength. Default is tuned for
            typical DPC data.
        max_iter: Number of ADMM iterations. Default is 400.

    Returns:
        Deconvolved image with same shape as input.

    Notes:
        The algorithm alternates between:
        1. Wiener-filtered deconvolution step in Fourier domain
        2. Total variation denoising step
    """
    z = y.copy()
    x = y.copy()

    m, n = y.shape

    # Precompute Fourier transforms
    Fk = np.fft.fft2(psf, s=(2 * m, 2 * n))
    Fy = np.fft.fft2(y, s=(2 * m, 2 * n))

    for iteration in range(max_iter):
        # Fourier transform of current estimate
        Fz = np.fft.fft2(z, s=(2 * m, 2 * n))

        # Wiener-filtered update in Fourier domain
        denom = np.abs(Fk) ** 2 + regul_param
        x = np.real(np.fft.ifft2((np.conj(Fk) * Fy + regul_param * Fz) / denom))
        x = x[:m, :n]

        # TV denoising step
        epsilon = 1e-2
        lam = 0.02
        iter_tv = 5
        z = _tv_denoise(x, epsilon, lam, iter_tv)

    # Correct one-pixel shift artifact
    out = np.zeros_like(x)
    out[1:, 1:] = x[:-1, :-1]

    return out
