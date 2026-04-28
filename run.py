"""
Main script for 3D Shepp-Logan phantom DPC simulation and reconstruction.

Usage:
    python scripts/run.py --size 128 --angles 180 [--photon-count 1000] [--plot-slice 64]
"""

import argparse
import os
import sys
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
    apply_hilbert_filter_projections,
    backproject,
    filtered_backproject,
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
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create phantom
    print(f"\n[1/6] Creating phantom ({size}^3)...")
    phantom, voxel_size = create_phantom(voxel_grid=size)
    print(f"      Voxel size: {voxel_size:.4f} units, mu range: [{phantom.min():.4f}, {phantom.max():.4f}]")
    
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
    diff_projections = apply_horizontal_derivative(projections)

    # 5. Hilbert filter + backprojection
    print(f"\n[5/6] Reconstruction with Hilbert filter and backprojection...")
    filtered = apply_hilbert_filter_projections(diff_projections)
    rec_hilbert_fbp = backproject(filtered, geo, cor=-0.5) / voxel_size  # Scale back to physical units

    # 6. Deconvolution methods + standard FBP with ramp filter
    print(f"\n[6/6] Deconvolution methods + Filtered Back Projection...")

    # Wiener deconvolution
    noise_var = wiener_v0
    wiener_results = Parallel(n_jobs=-1)(
        delayed(wiener_deconvolution)(diff_projections[:, a, :], noise_var=noise_var)
        for a in tqdm(range(diff_projections.shape[1]), desc="      Wiener deconv")
    )
    deconv_wiener = np.array(wiener_results)
    rec_wiener_fbp = filtered_backproject(deconv_wiener, geo, cor=0.0, angles_first=True) / voxel_size

    # TV deconvolution
    tv_reg_param = tv_reg  
    tv_results = Parallel(n_jobs=-1)(
        delayed(tv_deconvolution)(diff_projections[:, a, :], regul_param=tv_reg_param, max_iter=100)
        for a in tqdm(range(diff_projections.shape[1]), desc="      TV deconv")
    )
    deconv_tv = np.array(tv_results)
    rec_tv_fbp = filtered_backproject(deconv_tv, geo, cor=0.0, angles_first=True) / voxel_size

    # Sparse deconvolution
    sparse_reg_param = sparse_reg
    sparse_results = Parallel(n_jobs=-1)(
        delayed(deconv_sparse)(diff_projections[:, a, :], we=sparse_reg_param, max_iter=100)
        for a in tqdm(range(diff_projections.shape[1]), desc="      Sparse deconv")
    )
    deconv_sparse_result = np.array(sparse_results)
    rec_sparse_fbp = filtered_backproject(deconv_sparse_result, geo, cor=0.0, angles_first=True) / voxel_size     

    # Save results and generate plot
    save_results_and_generate_plot(
        phantom=phantom,
        projections=projections,
        orig_projections=orig_projections,
        diff_projections=diff_projections,
        rec_hilbert_fbp=rec_hilbert_fbp,
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
    )
    print("\n[Done]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Shepp-Logan phantom DPC simulation and reconstruction with ASTRA"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract parameters from config and ensure correct types
    size = int(config["phantom_params"]["voxel_grid"])
    angles = int(config["phantom_params"]["angles"])
    photon_count = float(config["noise"]["photon_count"])
    output_dir = str(config["output"]["path"])
    plot_slice = int(config["output"]["plot_slice"])
    tv_reg = float(config["deconv_params"]["tv_lambda"])
    sparse_reg = float(config["deconv_params"]["sparse_we"])
    wiener_v0 = float(config["deconv_params"]["wiener_v0"])

    main(
        size=size,
        angles=angles,
        photon_count=photon_count,
        output_dir=output_dir,
        plot_slice=plot_slice,
        tv_reg=tv_reg,
        sparse_reg=sparse_reg,
        wiener_v0=wiener_v0
    )
