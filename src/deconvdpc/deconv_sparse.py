from typing import Optional
import numpy as np
from scipy.signal import convolve2d


def _deconv_l2_weighted(
    I: np.ndarray,
    filt1: np.ndarray,
    we: float,
    max_iter: int = 200,
    weight_x: Optional[np.ndarray] = None,
    weight_y: Optional[np.ndarray] = None,
    weight_xx: Optional[np.ndarray] = None,
    weight_yy: Optional[np.ndarray] = None,
    weight_xy: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Weighted L2 deconvolution using conjugate gradient method.

    Solves the deconvolution problem with spatial regularization using
    conjugate gradient optimization.

    Args:
        I: Input image (2D ndarray).
        filt1: Point spread function (2D ndarray with odd dimensions).
        we: Regularization weight. Larger values enforce stronger smoothness.
        max_iter: Maximum number of iterations. Default is 200.
        weight_x, weight_y, weight_xx, weight_yy, weight_xy: Spatial weight maps.
            If None, default uniform weights are used.

    Returns:
        Deconvolved image with same spatial dimensions as input.
    """
    n, m = I.shape

    # Calculate filter half-size
    hfs1_x1 = (filt1.shape[1] - 1) // 2
    hfs1_x2 = filt1.shape[1] // 2
    hfs1_y1 = (filt1.shape[0] - 1) // 2
    hfs1_y2 = filt1.shape[0] // 2

    hfs_x1, hfs_x2 = hfs1_x1, hfs1_x2
    hfs_y1, hfs_y2 = hfs1_y1, hfs1_y2

    m_padded = m + hfs_x1 + hfs_x2
    n_padded = n + hfs_y1 + hfs_y2

    # Create mask for valid region
    mask = np.zeros((n_padded, m_padded))
    mask[hfs_y1 : n_padded - hfs_y2, hfs_x1 : m_padded - hfs_x2] = 1

    # Initialize weights if not provided
    if weight_x is None:
        weight_x = np.ones((n_padded, m_padded - 1))
        weight_y = np.ones((n_padded - 1, m_padded))
        weight_xx = np.zeros((n_padded, m_padded - 2))
        weight_yy = np.zeros((n_padded - 2, m_padded))
        weight_xy = np.zeros((n_padded - 1, m_padded - 1))

    # Pad input image
    x = np.pad(I, ((hfs_y1, hfs_y2), (hfs_x1, hfs_x2)), mode="edge")

    # Compute right-hand side
    b = convolve2d(x * mask, filt1, mode="same")

    # Define derivative filters
    dxf = np.array([[1, -1]])
    dyf = np.array([[1], [-1]])
    dyyf = np.array([[-1], [2], [-1]])
    dxxf = np.array([[-1, 2, -1]])
    dxyf = np.array([[-1, 1], [1, -1]])

    # Initialize operator
    Ax = convolve2d(
        convolve2d(x, np.flipud(np.fliplr(filt1)), mode="same") * mask,
        filt1,
        mode="same",
    )
    Ax += we * convolve2d(
        weight_x * convolve2d(x, np.flipud(np.fliplr(dxf)), mode="valid"), dxf
    )
    Ax += we * convolve2d(
        weight_y * convolve2d(x, np.flipud(np.fliplr(dyf)), mode="valid"), dyf
    )
    Ax += we * convolve2d(
        weight_xx * convolve2d(x, np.flipud(np.fliplr(dxxf)), mode="valid"), dxxf
    )
    Ax += we * convolve2d(
        weight_yy * convolve2d(x, np.flipud(np.fliplr(dyyf)), mode="valid"), dyyf
    )
    Ax += we * convolve2d(
        weight_xy * convolve2d(x, np.flipud(np.fliplr(dxyf)), mode="valid"), dxyf
    )

    r = b - Ax
    rho_1 = 0.0
    p = None

    # Conjugate gradient iterations
    for iteration in range(1, max_iter + 1):
        rho = np.sum(r**2)

        if iteration > 1 and rho_1 > 0:
            beta = rho / rho_1
            p = r + beta * p
        else:
            p = r

        # Apply operator to search direction
        Ap = convolve2d(
            convolve2d(p, np.flipud(np.fliplr(filt1)), mode="same") * mask,
            filt1,
            mode="same",
        )
        Ap += we * convolve2d(
            weight_x * convolve2d(p, np.flipud(np.fliplr(dxf)), mode="valid"), dxf
        )
        Ap += we * convolve2d(
            weight_y * convolve2d(p, np.flipud(np.fliplr(dyf)), mode="valid"), dyf
        )
        Ap += we * convolve2d(
            weight_xx * convolve2d(p, np.flipud(np.fliplr(dxxf)), mode="valid"), dxxf
        )
        Ap += we * convolve2d(
            weight_yy * convolve2d(p, np.flipud(np.fliplr(dyyf)), mode="valid"), dyyf
        )
        Ap += we * convolve2d(
            weight_xy * convolve2d(p, np.flipud(np.fliplr(dxyf)), mode="valid"), dxyf
        )

        alpha = rho / np.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rho_1 = rho

    return x


def deconv_sparse(
    I: np.ndarray,
    we: float = 1e-4,
    max_iter: int = 400,
    n_inner_iter: int = 2,
    psf: np.ndarray = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]]),
) -> np.ndarray:
    """Sparse (iterative shrinkage) deconvolution.

    Performs deconvolution with edge-preserving regularization using
    iteratively reweighted L2 minimization. The weights are adapted based
    on image gradients to preserve edges.

    Args:
        I: Input image (2D ndarray).
        psf: Point spread function (2D ndarray with odd dimensions).
        we: Regularization weight parameter. Default affects weight computation.
        max_iter: Maximum iterations. Default is 200.
        n_inner_iter: Number of inner weight refinement iterations. Default is 2.

    Returns:
        Deconvolved image with same shape as input.

    Notes:
        PSF will be negated before processing. The algorithm performs
        iterative reweighting based on image gradients to promote sparsity
        in the gradient domain.
    """
    # Negate PSF
    psf = psf * -1

    n, m = I.shape

    # Calculate filter half-size
    hfs1_x1 = (psf.shape[1] - 1) // 2
    hfs1_x2 = psf.shape[1] // 2
    hfs1_y1 = (psf.shape[0] - 1) // 2
    hfs1_y2 = psf.shape[0] // 2

    hfs_x1, hfs_x2 = hfs1_x1, hfs1_x2
    hfs_y1, hfs_y2 = hfs1_y1, hfs1_y2

    m_padded = m + hfs_x1 + hfs_x2
    n_padded = n + hfs_y1 + hfs_y2

    # Create mask
    mask = np.zeros((n_padded, m_padded))
    mask[hfs_y1 : n_padded - hfs_y2, hfs_x1 : m_padded - hfs_x2] = 1

    # Pad input
    I_padded = np.zeros((n_padded, m_padded))
    I_padded[hfs_y1 : n_padded - hfs_y2, hfs_x1 : m_padded - hfs_x2] = I
    x = I_padded.copy()

    # Define derivative filters
    dxf = np.array([[1, -1]])
    dyf = np.array([[1], [-1]])
    dyyf = np.array([[-1], [2], [-1]])
    dxxf = np.array([[-1, 2, -1]])
    dxyf = np.array([[-1, 1], [1, -1]])

    # Initial weights
    weight_x = np.ones((n_padded, m_padded - 1))
    weight_y = np.ones((n_padded - 1, m_padded))
    weight_xx = np.ones((n_padded, m_padded - 2))
    weight_yy = np.ones((n_padded - 2, m_padded))
    weight_xy = np.ones((n_padded - 1, m_padded - 1))

    # Initial deconvolution
    x = _deconv_l2_weighted(
        I_padded[hfs_y1 : n_padded - hfs_y2, hfs_x1 : m_padded - hfs_x2],
        psf,
        we,
        max_iter,
        weight_x,
        weight_y,
        weight_xx,
        weight_yy,
        weight_xy,
    )

    # Weight adaptation parameters
    w0 = 0.1
    exp_a = 0.8
    thr_e = 0.01

    # Iterative reweighting
    for iteration in range(n_inner_iter):
        # Compute gradients
        dy = convolve2d(x, np.fliplr(np.flipud(dyf)), mode="valid")
        dx = convolve2d(x, np.fliplr(np.flipud(dxf)), mode="valid")
        dyy = convolve2d(x, np.fliplr(np.flipud(dyyf)), mode="valid")
        dxx = convolve2d(x, np.fliplr(np.flipud(dxxf)), mode="valid")
        dxy = convolve2d(x, np.fliplr(np.flipud(dxyf)), mode="valid")

        # Update weights based on gradients
        weight_x = w0 * np.maximum(np.abs(dx), thr_e) ** (exp_a - 2)
        weight_y = w0 * np.maximum(np.abs(dy), thr_e) ** (exp_a - 2)
        weight_xx = 0.25 * w0 * np.maximum(np.abs(dxx), thr_e) ** (exp_a - 2)
        weight_yy = 0.25 * w0 * np.maximum(np.abs(dyy), thr_e) ** (exp_a - 2)
        weight_xy = 0.25 * w0 * np.maximum(np.abs(dxy), thr_e) ** (exp_a - 2)

        # Refine solution
        x = _deconv_l2_weighted(
            I_padded[hfs_y1 : n_padded - hfs_y2, hfs_x1 : m_padded - hfs_x2],
            psf,
            we,
            max_iter,
            weight_x,
            weight_y,
            weight_xx,
            weight_yy,
            weight_xy,
        )

    # Crop to original size and correct pixel shift
    x = x[hfs_y1 : n_padded - hfs_y2, hfs_x1 : m_padded - hfs_x2]
    x = np.concatenate((x[:, 1:], x[:, -1:]), axis=1)

    return x
