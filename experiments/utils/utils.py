import matplotlib.pyplot as plt
import numpy as np
from nilearn.image import crop_img, resample_to_img
from sklearn.preprocessing import scale


def crop_img_w_ref(input, ref=None):
    """Crop image to reference image."""
    print("Cropping image...")
    cropped = crop_img(input)
    if ref is not None:
        cropped = resample_to_img(cropped, ref, interpolation="nearest")
    return cropped


def get_contrasts():
    CONTRASTS = [
        ["LANGUAGE", 1, "MATH"],
        ["LANGUAGE", 2, "STORY"],
        ["LANGUAGE", 3, "MATH-STORY"],
        ["RELATIONAL", 1, "MATCH"],
        ["RELATIONAL", 2, "REL"],
        ["RELATIONAL", 3, "MATCH-REL"],
        ["SOCIAL", 1, "RANDOM"],
        ["SOCIAL", 2, "TOM"],
        ["SOCIAL", 6, "TOM-RANDOM"],
        ["EMOTION", 1, "FACES"],
        ["EMOTION", 2, "SHAPES"],
        ["EMOTION", 3, "FACES-SHAPES"],
        ["WM", 1, "2BK_BODY"],
        ["WM", 2, "2BK_FACE"],
        ["WM", 3, "2BK_PLACE"],
        ["WM", 4, "2BK_TOOL"],
        ["WM", 5, "0BK_BODY"],
        ["WM", 6, "0BK_FACE"],
        ["WM", 7, "0BK_PLACE"],
        ["WM", 8, "0BK_TOOL"],
        ["WM", 9, "2BK"],
        ["WM", 10, "0BK"],
        ["WM", 11, "2BK-0BK"],
        ["WM", 15, "BODY"],
        ["WM", 16, "FACE"],
        ["WM", 17, "PLACE"],
        ["WM", 18, "TOOL"],
        ["WM", 19, "BODY-AVG"],
        ["WM", 20, "FACE-AVG"],
        ["WM", 21, "PLACE-AVG"],
        ["WM", 22, "TOOL-AVG"],
        ["MOTOR", 1, "CUE"],
        ["MOTOR", 2, "LF"],
        ["MOTOR", 3, "LH"],
        ["MOTOR", 4, "RF"],
        ["MOTOR", 5, "RH"],
        ["MOTOR", 6, "T"],
        ["MOTOR", 7, "AVG"],
        ["MOTOR", 8, "CUE-AVG"],
        ["MOTOR", 9, "LF-AVG"],
        ["MOTOR", 10, "LH-AVG"],
        ["MOTOR", 11, "RF-AVG"],
        ["MOTOR", 12, "RH-AVG"],
        ["MOTOR", 13, "T-AVG"],
        ["GAMBLING", 1, "PUNISH"],
        ["GAMBLING", 2, "REWARD"],
        ["GAMBLING", 3, "PUNISH-REWARD"],
    ]

    return CONTRASTS


def memory_use(var):
    """Return the memory usage of a numpy variable in GB."""
    print(f"Memory usage in GBs: {var.nbytes / 1024**3:.2f}")


def compute_corr_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def plot_corr_matrices_across_contrasts(
    all_corrs, contrasts, vmin=-0.2, vmax=0.7, title=None, cmap="plasma", verbose=False
):
    n_cont = len(contrasts)
    all_corrs = np.squeeze(all_corrs)
    if n_cont == 1:
        all_corrs = np.expand_dims(all_corrs, 0)
    fig, axes = plt.subplots(1, n_cont, figsize=(5 * n_cont, 5), squeeze=True)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.78, wspace=0.05)

    for i in range(n_cont):
        corr = all_corrs[i]
        cont_name = contrasts[i] if n_cont > 1 else contrasts
        ax = axes[i] if n_cont > 1 else axes

        true_vmax = np.nanmax(corr)
        true_vmin = np.nanmin(corr)
        true_mean = np.nanmean(corr)

        scaled_pred_corr = scale(corr, axis=0)
        scaled_pred_corr = scale(scaled_pred_corr, axis=1)

        scale_level = (np.nanmax(scaled_pred_corr) - np.nanmin(scaled_pred_corr)) / (
            true_vmax - true_vmin
        )
        scaled_pred_corr = scaled_pred_corr / scale_level + true_mean

        diag_corr = np.nanmean(np.diag(corr))
        offdiag_corr = np.nanmean(corr[np.triu_indices(corr.shape[0], -1)])

        cax = ax.imshow(scaled_pred_corr, cmap=cmap, vmin=vmin, vmax=vmax)

        val_range = [vmin, (vmin + vmax) / 2, vmax]
        tick_range = [vmin, (vmin + vmax) / 2, vmax]
        cbar = fig.colorbar(cax, ticks=val_range, ax=ax)
        cbar.ax.set_yticklabels([("%.2f" % val) for val in tick_range])

        ax.set_title(
            "%s\nDiagonal corr = %.3f\nOff-diagonal corr = %.3f\nDifference = %.3f (%.2f%%)"
            % (
                cont_name,
                diag_corr,
                offdiag_corr,
                diag_corr - offdiag_corr,
                (diag_corr - offdiag_corr) * 100.0 / offdiag_corr,
            )
        )

        if verbose:
            print(
                "True min",
                true_vmin,
                " - True max",
                true_vmax,
                " - True mean",
                true_mean,
            )
            print(
                "Scaled min",
                np.nanmin(scaled_pred_corr),
                " - Scaled max",
                np.nanmax(scaled_pred_corr),
                " - Scaled mean = ",
                np.nanmean(scaled_pred_corr),
            )
            print(
                "Rescaled min",
                np.nanmin(scaled_pred_corr),
                " - Rescaled max",
                np.nanmax(scaled_pred_corr),
                " - Rescaled mean = ",
                np.nanmean(scaled_pred_corr),
            )

    if title is not None:
        fig.suptitle(title)
