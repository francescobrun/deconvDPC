import numpy as np


def wiener_deconvolution(
    image: np.ndarray,
    noise_var: float,
    psf: np.ndarray = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]]),
) -> np.ndarray:
    """Wiener deconvolution in Fourier domain.

    Applies Wiener filter for image deconvolution using Fourier transforms.
    Suitable for images with known noise characteristics.

    Args:
        image: Input image (2D ndarray).
        psf: Point spread function (2D ndarray).
        noise_var: Noise variance (Wiener filter parameter).

    Returns:
        Deconvolved image with same shape as input.

    Notes:
        The algorithm solves: X = (H^* / (|H|^2 + noise_var)) * Y
        where H is the PSF and Y is the input image in Fourier domain.
    """
    # Fourier transform
    F = np.fft.fft2(image)
    H = np.fft.fft2(psf, s=image.shape)

    # Wiener filter
    denom = np.abs(H) ** 2 + noise_var
    G = np.conj(H) / np.maximum(denom, np.finfo(np.float64).eps)
    X = G * F

    # Inverse Fourier transform
    x = np.fft.ifft2(X).real

    # Correct pixel shifts
    x = np.concatenate((x[:, :1], x[:, :-1]), axis=1)
    x = np.concatenate((x[:1, :], x[:-1, :]), axis=0)

    return x
