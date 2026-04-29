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
    profile_line: int = 20,
    profile_length: int = 64,
) -> None:

    print(f"\nSaving results to '{output_dir}'...")

    # Save TIFFs (unchanged)
    save_as_tiff(phantom, os.path.join(output_dir, "phantom.tiff"))
    save_as_tiff(projections, os.path.join(output_dir, "projections.tiff"))
    save_as_tiff(orig_projections, os.path.join(output_dir, "orig_projections.tiff"))
    save_as_tiff(diff_projections, os.path.join(output_dir, "diff_projections.tiff"))
    save_as_tiff(rec_hilbert_fbp, os.path.join(output_dir, "rec_hilbert_fbp.tiff"))
    save_as_tiff(deconv_wiener, os.path.join(output_dir, "deconvolved_wiener.tiff"))
    save_as_tiff(rec_wiener_fbp, os.path.join(output_dir, "recon_wiener_fbp.tiff"))
    save_as_tiff(deconv_tv, os.path.join(output_dir, "deconvolved_tv.tiff"))
    save_as_tiff(rec_tv_fbp, os.path.join(output_dir, "recon_tv_fbp.tiff"))
    save_as_tiff(deconv_sparse_result, os.path.join(output_dir, "deconvolved_sparse.tiff"))
    save_as_tiff(rec_sparse_fbp, os.path.join(output_dir, "recon_sparse_fbp.tiff"))

    try:
        import matplotlib.pyplot as plt
        import matplotlib.widgets as mwidgets

        mid_z = phantom.shape[0] // 2 if plot_slice is None else plot_slice
        mid_a = angles // 2

        hilbert_slice = rec_hilbert_fbp[mid_z]
        hilbert_vmin, hilbert_vmax = hilbert_slice.min(), hilbert_slice.max()

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))

        # Projection
        ax_proj = axes[0, 1]
        im_proj = ax_proj.imshow(diff_projections[:, mid_a, :], cmap='gray')
        ax_proj.set_title(f"Differential projection (angle {mid_a})")
        plt.colorbar(im_proj, ax=ax_proj, fraction=0.046, pad=0.04)

        volumes = {
            'phantom': (phantom, None, "Phantom"),
            'hilbert': (rec_hilbert_fbp, (hilbert_vmin, hilbert_vmax), "Hilbert + BP"),
            'wiener': (rec_wiener_fbp, (hilbert_vmin, hilbert_vmax), "Wiener + FBP"),
            'tv': (rec_tv_fbp, (hilbert_vmin, hilbert_vmax), "TV Deconv + FBP"),
            'sparse': (rec_sparse_fbp, (hilbert_vmin, hilbert_vmax), "Sparse + FBP"),
        }

        positions = [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]
        ims = {}
        phantom_ax = None

        current_slice = [mid_z]
        current_line = [profile_line]
        current_line_collection = [None]

        for i, (name, (vol, vrange, base_title)) in enumerate(volumes.items()):
            ax = axes[positions[i][0], positions[i][1]]

            kwargs = {"cmap": "gray"}
            if vrange is not None:
                kwargs["vmin"] = vrange[0]
                kwargs["vmax"] = vrange[1]

            im = ax.imshow(vol[mid_z], **kwargs)
            ims[name] = (im, base_title)

            ax.set_title(f"{base_title} (slice {mid_z})")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if name == 'phantom':
                phantom_ax = ax

        def compute_profile(vol, slice_idx):
            slice_data = vol[slice_idx]
            line = slice_data[current_line[0], :]
            center = len(line) // 2
            start = max(0, center - profile_length // 2)
            end = min(len(line), center + profile_length // 2)
            return line[start:end], start, end

        def draw_line(ax, vol, slice_idx):
            for line in ax.lines[:]:
                line.remove()

            if current_line_collection[0] is not None:
                current_line_collection[0].remove()

            slice_data = vol[slice_idx]
            line_data = slice_data[current_line[0], :]

            center = len(line_data) // 2
            start = max(0, center - profile_length // 2)
            end = min(len(line_data), center + profile_length // 2)

            line_coll = ax.hlines(
                current_line[0],
                start,
                end,
                colors='red',
                linewidth=2
            )
            current_line_collection[0] = line_coll

        draw_line(phantom_ax, phantom, mid_z)

        phantom_ax.text(0.5, -0.15, "Click to change the line position",
                        transform=phantom_ax.transAxes, ha='center',
                        fontsize=10, style='italic', color='gray')

        ax_profile = axes[0, 3]
        ax_profile.set_title("Line Profiles")

        colors = {
            'phantom': 'blue',
            'hilbert': 'green',
            'wiener': 'red',
            'tv': 'cyan',
            'sparse': 'magenta'
        }

        profiles = {}

        for name, (vol, _, _) in volumes.items():
            prof, start, end = compute_profile(vol, mid_z)
            x = np.arange(len(prof))
            line_plot, = ax_profile.plot(x, prof, color=colors[name], label=name)
            profiles[name] = line_plot

        ax_profile.legend()
        ax_profile.set_xlabel("Position")
        ax_profile.set_ylabel("Intensity")

        # Slider axes (position will be adjusted AFTER tight_layout)
        ax_slider = axes[1, 3]

        slider = mwidgets.Slider(
            ax_slider,
            'Slice',
            0,
            phantom.shape[0] - 1,
            valinit=mid_z,
            valstep=1,
            orientation='horizontal'
        )

        def update_all():
            slice_idx = current_slice[0]

            for name, (im, base_title) in ims.items():
                vol = volumes[name][0]
                im.set_data(vol[slice_idx])
                im.axes.set_title(f"{base_title} (slice {slice_idx})")
                profile, _, _ = compute_profile(vol, slice_idx)
                profiles[name].set_ydata(profile)

            draw_line(phantom_ax, phantom, slice_idx)
            fig.canvas.draw_idle()

        def on_slider(val):
            current_slice[0] = int(val)
            update_all()

        slider.on_changed(on_slider)

        def on_click(event):
            if event.inaxes != phantom_ax or event.ydata is None:
                return
            current_line[0] = int(event.ydata)
            update_all()

        fig.canvas.mpl_connect('button_press_event', on_click)

        update_all()

        plt.tight_layout()

        # ✅ Applied AFTER tight_layout() so it isn't overridden
        pos = ax_slider.get_position()
        new_height = pos.height * 0.25
        ax_slider.set_position([
            pos.x0,
            pos.y0 + (pos.height - new_height) / 2,
            pos.width,
            new_height
        ])

        plt.show()
        plt.close(fig)

    except ImportError:
        pass