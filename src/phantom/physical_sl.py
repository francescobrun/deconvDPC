import numpy as np
from ._sl3d import shepp_logan_3d

# ============================================================================
# Physical Shepp-Logan Phantom Generator
# ============================================================================
# This module generates 3D Shepp-Logan phantoms with physical dimensions
# for use in deconvolution and reconstruction testing


def create_phantom(voxel_grid=256, physical_FOV_cm=30.0):
    """
    Generate a 3D Shepp-Logan phantom with physical scaling information.

    The Shepp-Logan phantom is a standard test image used in medical imaging
    research. This function returns the normalized phantom (0-1 range) along
    with its voxel size, allowing the caller to apply physical scaling as needed
    in the imaging pipeline.

    Parameters
    ----------
    voxel_grid : int, optional
        Number of voxels along each dimension. Default is 256.
        Creates a cubic grid of size (voxel_grid, voxel_grid, voxel_grid).

    physical_FOV_cm : float, optional
        Physical field of view in centimeters. Default is 30.0 cm.
        Used to calculate the voxel size in the physical domain.

    Returns
    -------
    phantom : np.ndarray
        Normalized 3D Shepp-Logan phantom (range 0-1 representing tissue contrast).
        Shape: (voxel_grid, voxel_grid, voxel_grid)
        Data type: float32

    """
    # Generate the 3D Shepp-Logan phantom using the Yu-Ye-Wang variant
    # and convert to float32 for numerical computation
    phantom = shepp_logan_3d(size_out=voxel_grid).astype(np.float32)

    # Calculate the physical size of each voxel in centimeters
    voxel_size = physical_FOV_cm / voxel_grid

    return phantom, voxel_size
