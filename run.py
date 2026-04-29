"""
Main script for 3D Shepp-Logan phantom DPC simulation and reconstruction.

Usage:
    python scripts/run.py --size 128 --angles 180 [--photon-count 1000] [--plot-slice 64]
"""

import argparse
import os
import sys
import yaml
import yaml

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils.utils import save_as_tiff, save_results_and_generate_plot
from deconvdpc.deconv_sparse import deconv_sparse
from deconvdpc.deconv_wiener import wiener_deconvolution
from deconvdpc.deconv_tv import tv_deconvolution
from phantom.physical_sl import create_phantom
from recon.recon_astra import (
    add_poisson_noise,
    apply_horizontal_derivative,
    hilbert_filter,
    BP,
    FBP,
    forward_project,
)


def main(
    size: int = 256,
    angles: int = 360,
    photon_count: float = 10000.0,
    output_dir: str = "output",
    plot_slice: int | None = None,
    tv_reg: float = 1e-2,
    sparse_reg: float = 1e-3,
    wiener_v0: float = 1e-5,
    profile_line: int = 20,
    profile_length: int = 64,
) -> None:
    """Run the full DPC simulation and reconstruction pipeline.

    Args:
        size: Cubic phantom size in voxels.
        angles: Number of projection angles over [0, pi).
        photon_count: Average photon count per pixel for Poisson noise.
            ``None`` or <= 0 means noise-free.
        output_dir: Directory where output TIFF files are saved.
        plot_slice: Slice index to display in the overview figure. ``None``
            uses the middle slice.
        tv_reg: TV deconvolution regularization parameter.
        sparse_reg: Sparse deconvolution regularization parameter.
        wiener_v0: Wiener deconvolution noise variance.
        profile_line: Row index for line profile extraction.
        profile_length: Length of centered segment for line profile.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create phantom
    print(f"\n[1/6] Creating phantom ({size}^3)...")
    phantom, voxel_size = create_phantom(voxel_grid=size)
    print(
        f"      Voxel size: {voxel_size:.4f} cm, mu range: [{phantom.min():.4f}, {phantom.max():.4f}] cm^-1"
    )

    # 2. Forward projection
    print(f"\n[2/6] Forward projection ({angles} angles)...")
    angles_rad = np.linspace(0, np.pi, angles, endpoint=False).astype(np.float32)
    orig_projections, geo = forward_project(phantom, angles_rad)
    orig_projections = orig_projections * voxel_size

    # 3. Add Poisson noise
    print(f"\n[3/6] Adding Poisson noise (photons per pixel={photon_count})...")
    projections = add_poisson_noise(orig_projections, photon_count)

    # 4. Differential projections
    print(f"\n[4/6] Horizontal derivative...")
    diff_projs = apply_horizontal_derivative(projections)

    # 5. Hilbert filter + backprojection
    print(f"\n[5/6] Reconstruction with Hilbert filter and backprojection...")
    filtered = hilbert_filter(diff_projs)
    rec_hilbert_bp = (
        BP(filtered, geo, cor=-0.5) / voxel_size
    )  # Scale back to physical units

    # 6. Deconvolution methods + standard FBP with ramp filter
    print(f"\n[6/6] Deconvolution methods + Filtered Back Projection...")

    # Wiener deconvolution
    noise_var = wiener_v0
    wiener_results = Parallel(n_jobs=-2)(
        delayed(wiener_deconvolution)(diff_projs[:, a, :], noise_var=noise_var)
        for a in tqdm(range(diff_projs.shape[1]), desc="      Wiener deconv")
    )
    deconv_wiener = np.array(wiener_results)
    rec_wiener_fbp = FBP(deconv_wiener, geo, cor=0.0, angles_first=True) / voxel_size

    # TV deconvolution
    tv_reg_param = tv_reg
    tv_results = Parallel(n_jobs=-2)(
        delayed(tv_deconvolution)(
            diff_projs[:, a, :], regul_param=tv_reg_param, max_iter=100
        )
        for a in tqdm(range(diff_projs.shape[1]), desc="      TV deconv")
    )
    deconv_tv = np.array(tv_results)
    rec_tv_fbp = FBP(deconv_tv, geo, cor=0.0, angles_first=True) / voxel_size

    # Sparse deconvolution
    sparse_reg_param = sparse_reg
    sparse_results = Parallel(n_jobs=-2)(
        delayed(deconv_sparse)(
            diff_projs[:, a, :], we=sparse_reg_param, max_iter=100
        )
        for a in tqdm(range(diff_projs.shape[1]), desc="      Sparse deconv")
    )
    deconv_sparse_result = np.array(sparse_results)
    rec_sparse_fbp = (
        FBP(deconv_sparse_result, geo, cor=0.0, angles_first=True) / voxel_size
    )

    # Save results and generate plot
    save_results_and_generate_plot(
        phantom=phantom,
        projections=projections,
        orig_projections=orig_projections,
        diff_projections=diff_projs,
        rec_hilbert_fbp=rec_hilbert_bp,
        deconv_wiener=deconv_wiener,
        rec_wiener_fbp=rec_wiener_fbp,
        deconv_tv=deconv_tv,
        rec_tv_fbp=rec_tv_fbp,
        deconv_sparse_result=deconv_sparse_result,
        rec_sparse_fbp=rec_sparse_fbp,
        output_dir=output_dir,
        angles=angles,
        photon_count=photon_count,
        plot_slice=plot_slice,
        profile_line=profile_line,
        profile_length=profile_length,
    )
    print("\n[Done]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Shepp-Logan phantom DPC simulation and reconstruction"
    )
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    phantom_params = config['phantom_params']
    noise_params = config['noise']
    deconv_params = config['deconv_params']
    output_params = config['output']
    plot_params = config.get('plot_params', {})

    main(
        size=phantom_params['voxel_grid'],
        angles=phantom_params['angles'],
        photon_count=float(noise_params['photon_count']),
        output_dir=output_params['path'],
        plot_slice=output_params.get('plot_slice'),
        tv_reg=float(deconv_params['tv_lambda']),
        sparse_reg=float(deconv_params['sparse_we']),
        wiener_v0=float(deconv_params['wiener_v0']),
        profile_line=plot_params.get('profile_line', 20),
        profile_length=plot_params.get('profile_length', 64),
    )