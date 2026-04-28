import os
from typing import Optional

import numpy as np
import tifffile


def save_as_tiff(volume: np.ndarray, filepath: str) -> None:
    """Save a 3D volume as a multi-page TIFF file.

    Each slice along the first axis is saved as an individual page.
    The output dtype is ``float32`` to preserve numerical precision.

    Args:
        volume: 3D array with shape ``(nz, ny, nx)``.
        filepath: Output file path. The ``.tif`` or ``.tiff`` extension is
            appended automatically if missing.

    Examples:
        >>> import numpy as np
        >>> vol = np.random.rand(10, 64, 64).astype(np.float32)
        >>> save_as_tiff(vol, "reconstruction.tiff")
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    if not filepath.lower().endswith((".tif", ".tiff")):
        filepath = filepath + ".tif"

    os.makedirs(
        os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True
    )
    tifffile.imwrite(
        filepath, volume.astype(np.float32), imagej=True, metadata={"axes": "ZYX"}
    )


def perc_norm(
    source: np.ndarray, reference: np.ndarray, p_low: int = 1, p_high: int = 99
) -> np.ndarray:
    """Normalize source image to match reference image contrast range.

    Rescales the intensity values of the source image to match the contrast
    range of the reference image using percentile-based normalization.

    Args:
        source: Source image to normalize (ndarray).
        reference: Reference image defining target contrast range (ndarray).
        p_low: Lower percentile threshold. Default is 1.
        p_high: Upper percentile threshold. Default is 99.

    Returns:
        Normalized source image with same shape and dtype as source.

    Examples:
        >>> source = np.random.rand(256, 256)
        >>> reference = np.random.rand(256, 256) * 2
        >>> normalized = perc_norm(source, reference)
        >>> print(normalized.shape)
        (256, 256)
    """
    src_low, src_high = np.percentile(source, [p_low, p_high])
    ref_low, ref_high = np.percentile(reference, [p_low, p_high])

    # Normalize source to [0, 1]
    source_norm = (source - src_low) / np.maximum(src_high - src_low, 1e-10)
    # Scale to reference range
    source_scaled = source_norm * (ref_high - ref_low) + ref_low

    return source_scaled


def save_results_and_generate_plot(
    phantom: np.ndarray,
    projections: np.ndarray,
    orig_projections: np.ndarray,
    diff_projections: np.ndarray,
    rec_hilbert_fbp: np.ndarray,
    deconv_wiener: np.ndarray,
    rec_wiener_fbp: np.ndarray,
    deconv_tv: np.ndarray,
    rec_tv_fbp: np.ndarray,
    deconv_sparse_result: np.ndarray,
    rec_sparse_fbp: np.ndarray,
    output_dir: str,
    angles: int,
    photon_count: Optional[float] = None,
    plot_slice: Optional[int] = None,
) -> None:
    """
    Save all volume results as TIFF files and generate the overview comparison plot.

    This function encapsulates all output saving logic:
    - Saves all intermediate and final volumes as TIFF files
    - Generates the comparative plot with PSNR/NRMSE metrics
    - Handles matplotlib import safely

    Args:
        phantom: Ground truth 3D phantom volume
        projections: Noisy projection data
        orig_projections: Original noise-free projection data
        diff_projections: Horizontal derivative projections
        rec_hilbert_fbp: Hilbert + backprojection reconstruction
        deconv_wiener: Wiener deconvolved projections
        rec_wiener_fbp: Wiener deconvolution reconstruction
        deconv_tv: TV deconvolved projections
        rec_tv_fbp: TV deconvolution reconstruction
        deconv_sparse_result: Sparse deconvolved projections
        rec_sparse_fbp: Sparse deconvolution reconstruction
        output_dir: Directory where outputs will be saved
        angles: Number of projection angles used
        photon_count: Photon count for Poisson noise (if used)
        plot_slice: Slice index to use for the plot (defaults to middle slice)
    """
    print(f"\nSaving results to '{output_dir}'...")

    # Save all volumes as TIFF
    save_as_tiff(phantom, os.path.join(output_dir, "phantom.tiff"))
    save_as_tiff(projections, os.path.join(output_dir, "projections.tiff"))
    save_as_tiff(orig_projections, os.path.join(output_dir, "orig_projections.tiff"))
    save_as_tiff(diff_projections, os.path.join(output_dir, "diff_projections.tiff"))
    save_as_tiff(rec_hilbert_fbp, os.path.join(output_dir, "rec_hilbert_fbp.tiff"))
    save_as_tiff(deconv_wiener, os.path.join(output_dir, "deconvolved_wiener.tiff"))
    save_as_tiff(rec_wiener_fbp, os.path.join(output_dir, "recon_wiener_fbp.tiff"))
    save_as_tiff(deconv_tv, os.path.join(output_dir, "deconvolved_tv.tiff"))
    save_as_tiff(rec_tv_fbp, os.path.join(output_dir, "recon_tv_fbp.tiff"))
    save_as_tiff(
        deconv_sparse_result, os.path.join(output_dir, "deconvolved_sparse.tiff")
    )
    save_as_tiff(rec_sparse_fbp, os.path.join(output_dir, "recon_sparse_fbp.tiff"))

    # Generate overview plot
    try:
        import matplotlib.pyplot as plt

        mid_z = phantom.shape[0] // 2 if plot_slice is None else plot_slice
        mid_a = angles // 2

        phantom_slice = phantom[mid_z]

        # Use Hilbert reconstruction range for consistent visualization
        hilbert_slice = rec_hilbert_fbp[mid_z]
        hilbert_vmin, hilbert_vmax = hilbert_slice.min(), hilbert_slice.max()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        plot_configs = [
            (axes[0, 0], phantom[mid_z], f"Phantom (slice {mid_z})", None),
            (
                axes[0, 1],
                diff_projections[:, mid_a, :],
                f"Differential projection (angle {mid_a})",
                None,
            ),
            (
                axes[0, 2],
                rec_hilbert_fbp[mid_z],
                f"Hilbert + BP",
                (hilbert_vmin, hilbert_vmax),
            ),
            (
                axes[1, 0],
                rec_wiener_fbp[mid_z],
                f"Wiener + FBP",
                (hilbert_vmin, hilbert_vmax),
            ),
            (
                axes[1, 1],
                rec_tv_fbp[mid_z],
                f"TV Deconv + FBP",
                (hilbert_vmin, hilbert_vmax),
            ),
            (
                axes[1, 2],
                rec_sparse_fbp[mid_z],
                f"Sparse + FBP",
                (hilbert_vmin, hilbert_vmax),
            ),
        ]

        for ax, arr, title, vrange in plot_configs:
            kwargs = {"cmap": "gray"}
            if vrange is not None:
                kwargs["vmin"] = vrange[0]
                kwargs["vmax"] = vrange[1]
            im = ax.imshow(arr, **kwargs)
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(
            f"3D Shepp-Logan DPC — slice {mid_z}, {angles} angles"
            + (f", photons={photon_count}" if photon_count else ", noise-free"),
            fontsize=12,
        )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "output_plot.png"), dpi=150)
        print("      Saved output_plot.png")
        plt.show()
        plt.close(fig)

    except ImportError:
        pass
