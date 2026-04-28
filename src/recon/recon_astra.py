"""ASTRA-based forward projection, DPC filtering, and backprojection utilities."""

import numpy as np
from scipy.signal import hilbert

try:
    import astra
except ImportError as exc:
    raise ImportError(
        "ASTRA Toolbox is required but not installed. "
        "Install it with: conda install -c astra-toolbox astra-toolbox"
    ) from exc


def add_poisson_noise(projections, photon_count=None):
    """Add realistic Poisson noise to projection data (simulating photon counting).

    Converts attenuation coefficient projections to photon counts via Beer-Lambert law,
    applies Poisson noise, then converts back to attenuation space. A realistic
    minimum photon floor prevents unphysical extreme values from zero-count events.

    Args:
        projections: Attenuation coefficient data (positive values).
        photon_count: Mean incident photon count per detector pixel.

    Returns:
        Noisy attenuation projections in the same units as input.
    """
    if photon_count is None or photon_count <= 0:
        return projections.copy()

    # Beer–Lambert law: convert attenuation to transmission
    T = np.exp(-projections)

    # Expected photon counts
    I = photon_count * T

    # Simulate photon counting noise (Poisson distribution)
    I_noise = np.random.poisson(I).astype(np.float32)

    # Prevent zero counts which lead to infinite attenuation; set a floor at 1 photon
    T_noise = np.maximum(I_noise, 1.0) / photon_count

    # Convert back to attenuation coefficient
    att_noise = -np.log(T_noise)

    return att_noise


def apply_horizontal_derivative(projections: np.ndarray) -> np.ndarray:
    """Apply a horizontal derivative kernel to projection data.

    The derivative is computed along the detector column direction (last axis)
    using a 3x3 kernel:
    [[0.0, 0.0, 0.0],
     [0.0, -1.0, 1.0],
     [0.0, 0.0, 0.0]]

    Args:
        projections: Array of shape ``(n_angles, det_row, det_col)``.

    Returns:
        Differential projections with the same shape as the input.
    """
    from scipy.ndimage import convolve

    kernel = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
    diff = convolve(projections, kernel[np.newaxis, :, :], mode="constant")

    return diff


def apply_hilbert_filter_projections(
    projections: np.ndarray, invert_sign: bool = True
) -> np.ndarray:
    """Apply the Hilbert-integration filter along detector columns.

    For DPC data that is the spatial derivative of the attenuation projection,
    the FBP filter that compensates both the derivative *and* the 1/r
    backprojection blur is

        ``G(f) = -j·sgn(f) / (2π)``

    which is the Hilbert transform scaled by ``1/(2π)``.

    Args:
        projections: Differential projection data of shape
            ``(n_angles, det_row, det_col)``.
        invert_sign: If True, invert the output sign to compensate for inverted
            gray levels / negated projection data. Use this when working with
            inverted intensity data.

    Returns:
        Filtered projections ready for backprojection.
    """
    # scipy.signal.hilbert returns the analytic signal z = x + j·H{x}
    # The Hilbert transform H{x} is the imaginary part.
    analytic = hilbert(projections, axis=2)

    sign = -1.0 if invert_sign else 1.0
    filtered = sign * analytic.imag / (2.0 * np.pi)

    return filtered


def forward_project(
    phantom: np.ndarray,
    angles: np.ndarray,
    det_spacing: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Perform 3D parallel-beam forward projection with ASTRA.

    Args:
        phantom: 3D volume array with shape ``(nz, ny, nx)``.
        angles: 1D array of projection angles in radians.
        det_spacing: Detector pixel spacing (default 1.0).

    Returns:
        A tuple ``(projections, geometry_info)`` where ``projections`` has shape
        ``(nz, n_angles, nx)`` (ASTRA ``parallel3d`` convention) and
        ``geometry_info`` is a dict with ASTRA geometry objects and IDs that can
        be passed to ``backproject``.
    """
    nz, ny, nx = phantom.shape

    vol_geom = astra.create_vol_geom(ny, nx, nz)
    proj_geom = astra.create_proj_geom(
        "parallel3d", det_spacing, det_spacing, ny, nx, angles
    )

    vol_id = astra.data3d.create("-vol", vol_geom, phantom)
    proj_id = astra.data3d.create("-proj3d", proj_geom)

    cfg = astra.astra_dict("FP3D_CUDA")
    cfg["VolumeDataId"] = vol_id
    cfg["ProjectionDataId"] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    projections = astra.data3d.get(proj_id)

    # Clean up ASTRA objects immediately; geometry dicts are kept for backprojection
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(vol_id)
    astra.data3d.delete(proj_id)

    geometry_info = {
        "vol_geom": vol_geom,
        "proj_geom": proj_geom,
    }
    return projections, geometry_info


def backproject(
    projections: np.ndarray,
    geometry_info: dict,
    cor: float = 0.0,
) -> np.ndarray:
    """Perform 3D parallel-beam backprojection with ASTRA.

    A hard-coded post-alignment shift of 1 pixel is applied to correct for
    the centre-of-rotation offset.

    Args:
        projections: Projection data array with shape ``(nz, n_angles, nx)``.
        geometry_info: Dictionary returned by ``forward_project`` containing
            the ASTRA geometries.
        cor: Centre-of-rotation offset in pixels.

    Returns:
        Reconstructed 3D volume with shape ``(nz, ny, nx)``.
    """
    vol_geom = geometry_info["vol_geom"]
    proj_geom = geometry_info["proj_geom"]

    # Apply a 1-pixel COR correction
    proj_geom = astra.geom_postalignment(proj_geom, cor)

    rec_id = astra.data3d.create("-vol", vol_geom)
    proj_id = astra.data3d.create("-proj3d", proj_geom, projections)

    cfg = astra.astra_dict("BP3D_CUDA")
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectionDataId"] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    reconstruction = astra.data3d.get(rec_id)

    # Standard parallel-beam FBP normalization factor (π / n_angles)
    n_angles = projections.shape[1]
    reconstruction *= np.pi / n_angles

    # Apply circular mask to zero out values outside the maximum inscribed circle
    # reduced in diameter by 2 pixels (radius reduced by 1 pixel)
    nz, ny, nx = reconstruction.shape
    center_y, center_x = ny / 2.0, nx / 2.0
    max_radius = min(ny, nx) / 2.0 - 1.0  # Reduce radius by 1 pixel

    yy, xx = np.ogrid[:ny, :nx]
    dist_from_center = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    mask = dist_from_center <= max_radius

    # Apply mask to all z-slices
    for z in range(nz):
        reconstruction[z][~mask] = 0

    # Clean up local ASTRA objects
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)

    return reconstruction


def filtered_backproject(
    projections: np.ndarray,
    geometry_info: dict,
    cor: float = 0.0,
    angles_first: bool = False,
) -> np.ndarray:
    """Perform 3D parallel-beam Filtered Back Projection.

    Applies standard ramp filter in frequency domain then uses ASTRA backprojection.

    Args:
        projections: Projection data array. Expected shape is ``(nz, n_angles, nx)``
            unless ``angles_first=True``, in which case ``(n_angles, nz, nx)``.
        geometry_info: Dictionary returned by ``forward_project`` containing
            the ASTRA geometries.
        cor: Centre-of-rotation offset in pixels.
        angles_first: If True, input is assumed to be in shape ``(n_angles, nz, nx)``
            and will be transposed to ``(nz, n_angles, nx)`` automatically.

    Returns:
        Reconstructed 3D volume with shape ``(nz, ny, nx)``.
    """
    # Transpose if input is in (angles, height, width) order
    if angles_first:
        projections = projections.transpose(1, 0, 2)

    n = projections.shape[2]
    # Create frequency axis
    f = np.fft.fftfreq(n)
    # Standard ramp filter |f| for parallel beam FBP
    ramp = np.abs(f)

    # Apply filter along detector columns for all angles and rows
    proj_fft = np.fft.fft(projections, axis=2)
    proj_fft *= ramp[np.newaxis, np.newaxis, :]
    filtered_projections = np.fft.ifft(proj_fft, axis=2).real

    # Use existing backprojection which already includes correct normalization
    return backproject(filtered_projections, geometry_info, cor=cor)
