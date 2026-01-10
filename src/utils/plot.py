import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.segmentation import find_boundaries
from skimage import color
from sklearn.metrics import roc_curve, roc_auc_score

def setup_style():
    """Applies the IEEE-like plotting style used in the original script."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

# =========================================================
#  QUANTITATIVE PLOTS (Takes DataFrames)
# =========================================================

def median_iqr(vals):
    """
    Computes median and Interquartile Range (IQR) for a list of values.
    
    Args:
        vals (list or np.ndarray): Input values.
        
    Returns:
        tuple: (Median, Q1, Q3). Returns NaNs if input is empty.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    med = np.median(vals)
    q1  = np.percentile(vals, 25)
    q3  = np.percentile(vals, 75)
    return med, q1, q3

def corr_and_ba(ref, pred):
    """
    Computes Pearson correlation and Bland-Altman statistics.
    
    Args:
        ref (list): Reference measurements.
        pred (list): Predicted measurements.
        
    Returns:
        tuple: (r, bias, lower_LOA, upper_LOA, ref_valid, pred_valid)
    """
    ref  = np.asarray(ref,  dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(ref) & np.isfinite(pred)
    ref, pred = ref[mask], pred[mask]

    if ref.size < 2:
        return np.nan, np.nan, np.nan, np.nan, ref, pred

    r = np.corrcoef(ref, pred)[0, 1]
    diff = pred - ref
    bias = diff.mean()
    loa  = 1.96 * diff.std(ddof=1) if diff.size > 1 else 0.0
    lower = bias - loa
    upper = bias + loa
    return r, bias, lower, upper, ref, pred

def dice_reliability_curve(dice_values, n_points=101):
    """
    Computes the fraction of cases exceeding various Dice thresholds.
    
    Args:
        dice_values (list): List of Dice scores.
        n_points (int): Number of thresholds to evaluate between 0 and 1.
        
    Returns:
        tuple: (thresholds, coverage_fractions)
    """
    dice_values = np.asarray(dice_values, dtype=float)
    dice_values = dice_values[np.isfinite(dice_values)]
    t = np.linspace(0.0, 1.0, n_points)
    cov = [(dice_values >= thr).mean() for thr in t]
    return t, np.asarray(cov)

def plot_metrics_summary(df, save_path=None):
    """
    Plots Median Dice and HD95 with IQRs and distributions boxplots.
    
    Args:
        df (pd.DataFrame): DataFrame containing metrics.
        save_path (str): Optional path to save the figure.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    setup_style()
    structures = ["LV", "MYO", "LA"]
    
    dice_medians, dice_err = [], []
    hd_medians,   hd_err   = [], []
    
    dice_data = [] 
    hd_data   = [] 

    for s in structures:
        # Dice
        d_col = f"dice_{s}"
        if d_col in df.columns:
            vals = df[d_col].values
            d_med, d_q1, d_q3 = median_iqr(vals)
            dice_medians.append(d_med)
            dice_err.append((d_med - d_q1, d_q3 - d_med))
            dice_data.append(vals[np.isfinite(vals)])
        else:
            dice_medians.append(0)
            dice_err.append((0, 0))
            dice_data.append([])

        # HD95
        h_col = f"hd95_{s}"
        if h_col in df.columns:
            vals = df[h_col].values
            h_med, h_q1, h_q3 = median_iqr(vals)
            hd_medians.append(h_med)
            hd_err.append((h_med - h_q1, h_q3 - h_med))
            hd_data.append(vals[np.isfinite(vals)])
        else:
            hd_medians.append(0)
            hd_err.append((0, 0))
            hd_data.append([])

    dice_err = np.array(dice_err).T
    hd_err   = np.array(hd_err).T

    # Setup Figure
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), dpi=300)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    x = np.arange(len(structures))
    bar_width = 0.6
    
    colors_map = {"LV": "#1f77b4", "MYO": "#2ca02c", "LA": "#d62728"}
    bar_colors = [colors_map.get(s, "#888888") for s in structures]

    # Panel A: Median Dice per structure with IQR
    ax_a.bar(x, dice_medians, yerr=dice_err, width=bar_width, capsize=3,
             linewidth=0.7, edgecolor="k", color=bar_colors)
    
    for i, s in enumerate(structures):
        if not np.isnan(dice_medians[i]):
            ax_a.text(x[i], dice_medians[i] + dice_err[1, i] + 0.005,
                      f"{dice_medians[i]:.3f}", ha="center", va="bottom", fontsize=8)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(structures)
    ax_a.set_ylim(0.70, 1.00)
    ax_a.set_ylabel("Dice coefficient")
    ax_a.set_title("Median Dice per structure (IQR)")

    # Panel B: Median HD95 per structure with IQR
    ax_b.bar(x, hd_medians, yerr=hd_err, width=bar_width, capsize=3,
             linewidth=0.7, edgecolor="k", color=bar_colors)

    for i, s in enumerate(structures):
        if not np.isnan(hd_medians[i]):
            ax_b.text(x[i], hd_medians[i] + hd_err[1, i] + 0.5,
                      f"{hd_medians[i]:.2f}", ha="center", va="bottom", fontsize=8)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(structures)
    ax_b.set_ylabel("HD95 (pixels)")
    ax_b.set_title("Median Hausdorff distance (IQR)")

    # Panel C: Dice distributions
    box_c = ax_c.boxplot(dice_data, positions=x, widths=0.55, patch_artist=True,
                         medianprops=dict(linewidth=1.0, color="k"),
                         boxprops=dict(linewidth=0.7),
                         whiskerprops=dict(linewidth=0.7),
                         capprops=dict(linewidth=0.7))
    for patch, s in zip(box_c["boxes"], structures):
        patch.set_facecolor(colors_map.get(s, "#bbbbbb"))

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(structures)
    ax_c.set_ylim(0.70, 1.00)
    ax_c.set_ylabel("Dice coefficient")
    ax_c.set_title("Dice distribution across test set")

    # Panel D: HD95 distributions
    box_d = ax_d.boxplot(hd_data, positions=x, widths=0.55, patch_artist=True,
                         medianprops=dict(linewidth=1.0, color="k"),
                         boxprops=dict(linewidth=0.7),
                         whiskerprops=dict(linewidth=0.7),
                         capprops=dict(linewidth=0.7))
    for patch, s in zip(box_d["boxes"], structures):
        patch.set_facecolor(colors_map.get(s, "#bbbbbb"))

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(structures)
    ax_d.set_ylabel("HD95 (pixels)")
    ax_d.set_title("Hausdorff distance distribution")

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, lab in zip(axes.ravel(), panel_labels):
        ax.text(-0.05, 1.1, lab, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="top")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    return fig

def plot_clinical_bland_altman(df_ef, save_path=None):
    """
    Plots Correlation and Bland-Altman analysis for ED, ES, and EF.
    
    Args:
        df_ef (pd.DataFrame): DataFrame containing clinical metrics.
        save_path (str): Optional path to save the figure.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    setup_style()
    
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), dpi=300)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()
    fig.subplots_adjust(hspace=0.4, wspace=0.35)
    
    marker_kw = dict(s=12, alpha=0.7, edgecolor="none")

    # ED Analysis
    r_ed, b_ed, l_ed, u_ed, ED_ref, ED_pred = corr_and_ba(df_ef["ED_ref"], df_ef["ED_pred"])
    
    if len(ED_ref) > 0:
        min_ed, max_ed = min(ED_ref.min(), ED_pred.min()), max(ED_ref.max(), ED_pred.max())
        ax_a.scatter(ED_ref, ED_pred, **marker_kw)
        ax_a.plot([min_ed, max_ed], [min_ed, max_ed], "k--", linewidth=0.8)
    ax_a.set_xlabel("Reference ED area (pixels)")
    ax_a.set_ylabel("Predicted ED area (pixels)")
    ax_a.set_title("ED area")

    if len(ED_ref) > 0:
        mean_ed = 0.5 * (ED_ref + ED_pred)
        diff_ed = ED_pred - ED_ref
        ax_d.scatter(mean_ed, diff_ed, **marker_kw)
        ax_d.axhline(b_ed, color="k", linewidth=1.0, label="Bias")
        ax_d.axhline(l_ed, color="k", linestyle="--", linewidth=0.8, label="LOA")
        ax_d.axhline(u_ed, color="k", linestyle="--", linewidth=0.8)
    ax_d.set_xlabel("Mean ED area (pixels)")
    ax_d.set_ylabel("Pred - Ref")
    ax_d.set_title("ED area Bland-Altman")

    # ES Analysis
    r_es, b_es, l_es, u_es, ES_ref, ES_pred = corr_and_ba(df_ef["ES_ref"], df_ef["ES_pred"])

    if len(ES_ref) > 0:
        min_es, max_es = min(ES_ref.min(), ES_pred.min()), max(ES_ref.max(), ES_pred.max())
        ax_b.scatter(ES_ref, ES_pred, **marker_kw)
        ax_b.plot([min_es, max_es], [min_es, max_es], "k--", linewidth=0.8)
    ax_b.set_xlabel("Reference ES area (pixels)")
    ax_b.set_ylabel("Predicted ES area (pixels)")
    ax_b.set_title("ES area")

    if len(ES_ref) > 0:
        mean_es = 0.5 * (ES_ref + ES_pred)
        diff_es = ES_pred - ES_ref
        ax_e.scatter(mean_es, diff_es, **marker_kw)
        ax_e.axhline(b_es, color="k", linewidth=1.0)
        ax_e.axhline(l_es, color="k", linestyle="--", linewidth=0.8)
        ax_e.axhline(u_es, color="k", linestyle="--", linewidth=0.8)
    ax_e.set_xlabel("Mean ES area (pixels)")
    ax_e.set_ylabel("Pred - Ref")
    ax_e.set_title("ES area Bland-Altman")

    # EF Analysis
    r_ef, b_ef, l_ef, u_ef, EF_ref, EF_pred = corr_and_ba(df_ef["EF_ref"], df_ef["EF_pred"])

    if len(EF_ref) > 0:
        min_ef, max_ef = min(EF_ref.min(), EF_pred.min()), max(EF_ref.max(), EF_pred.max())
        ax_c.scatter(EF_ref, EF_pred, **marker_kw)
        ax_c.plot([min_ef, max_ef], [min_ef, max_ef], "k--", linewidth=0.8)
    ax_c.set_xlabel("Reference EF (%)")
    ax_c.set_ylabel("Predicted EF (%)")
    ax_c.set_title("EF")

    if len(EF_ref) > 0:
        mean_ef = 0.5 * (EF_ref + EF_pred)
        diff_ef = EF_pred - EF_ref
        ax_f.scatter(mean_ef, diff_ef, **marker_kw)
        ax_f.axhline(b_ef, color="k", linewidth=1.0)
        ax_f.axhline(l_ef, color="k", linestyle="--", linewidth=0.8)
        ax_f.axhline(u_ef, color="k", linestyle="--", linewidth=0.8)
    ax_f.set_xlabel("Mean EF (%)")
    ax_f.set_ylabel("Pred - Ref")
    ax_f.set_title("EF Bland-Altman")

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for ax, label in zip(axes.ravel(), panel_labels):
        ax.text(-0.3, 1.12, label, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="top")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig

def plot_reliability_curves(df_metrics, df_ef, save_path=None):
    """
    Plots Dice and EF reliability curves.
    
    Args:
        df_metrics (pd.DataFrame): Dice metrics.
        df_ef (pd.DataFrame): EF metrics.
        save_path (str): Optional save path.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    setup_style()
    
    # Prepare Data
    t_all, cov_LV = dice_reliability_curve(df_metrics.get("dice_LV", []))
    _,    cov_MYO = dice_reliability_curve(df_metrics.get("dice_MYO", []))
    _,     cov_LA = dice_reliability_curve(df_metrics.get("dice_LA", []))
    
    if "EF_abs_err" in df_ef.columns:
        ef_abs = df_ef["EF_abs_err"].dropna().values
    else:
        ef_abs = np.abs(df_ef["EF_pred"] - df_ef["EF_ref"]).dropna().values

    ef_thr = np.array([1, 2, 3, 4, 5, 7.5, 10, 15, 20], dtype=float)
    ef_cov = np.array([(ef_abs <= thr).mean() for thr in ef_thr])

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), dpi=300)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    colors_map = {"LV": "#1f77b4", "MYO": "#2ca02c", "LA": "#d62728"}

    # Panel A: Dice reliability (all)
    ax_a.plot(t_all, cov_LV,  label="LV",  color=colors_map["LV"],  lw=1.5)
    ax_a.plot(t_all, cov_MYO, label="MYO", color=colors_map["MYO"], lw=1.5)
    ax_a.plot(t_all, cov_LA,  label="LA",  color=colors_map["LA"],  lw=1.5)
    ax_a.set_xlabel("Dice threshold τ")
    ax_a.set_ylabel("Fraction Dice ≥ τ")
    ax_a.set_xlim(0.0, 1.0); ax_a.set_ylim(0.0, 1.05)
    ax_a.set_title("Dice reliability (all)")
    ax_a.legend(frameon=False, fontsize=8, loc="lower left")

    # Panel B: Zoomed Dice reliability
    mask_zoom = (t_all >= 0.80)
    if mask_zoom.any():
        ax_b.plot(t_all[mask_zoom], cov_LV[mask_zoom],  color=colors_map["LV"],  lw=1.5)
        ax_b.plot(t_all[mask_zoom], cov_MYO[mask_zoom], color=colors_map["MYO"], lw=1.5)
        ax_b.plot(t_all[mask_zoom], cov_LA[mask_zoom],  color=colors_map["LA"],  lw=1.5)
    ax_b.set_xlabel("Dice threshold τ")
    ax_b.set_ylabel("Fraction Dice ≥ τ")
    ax_b.set_xlim(0.80, 1.0); ax_b.set_ylim(0.0, 1.05)
    ax_b.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax_b.set_title("Dice reliability (high-quality)")

    # Panel C: EF Error Histogram
    if len(ef_abs) > 0:
        bins = max(10, min(30, int(np.sqrt(len(ef_abs)))))
        ax_c.hist(ef_abs, bins=bins, color="#9467bd", edgecolor="k", alpha=0.85)
        
        for thr in [5, 10]:
            ax_c.axvline(thr, color="k", linestyle="--", linewidth=0.8)
            ax_c.text(thr, ax_c.get_ylim()[1]*0.85, f"{thr}%", rotation=90, va="top", ha="right", fontsize=7)
    
    ax_c.set_xlabel("|EF error| (%)")
    ax_c.set_ylabel("Count")
    ax_c.set_title("Distribution of |EF error|")

    # Panel D: EF Reliability Curve
    ax_d.step(ef_thr, ef_cov, where="post", color="#9467bd", lw=1.5, marker="o", markersize=3)
    ax_d.set_xlabel("|EF error| threshold (%)")
    ax_d.set_ylabel("Fraction |EF err| ≤ thr")
    ax_d.set_ylim(0.0, 1.05)
    ax_d.set_xlim(ef_thr[0], ef_thr[-1])
    ax_d.set_title("EF reliability curve")

    for x, y in zip(ef_thr, ef_cov):
        ax_d.text(x, y + 0.03, f"{y*100:.0f}%", ha="center", va="bottom", fontsize=6)

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, lab in zip(axes.ravel(), panel_labels):
        ax.text(-0.14, 1.08, lab, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="top")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig

def _overlay_contours(ax, img, gt, pr, lv_label=1):
    """
    Helper to draw contours on an axis.
    """
    img_n = (img - img.min()) / (img.max() - img.min() + 1e-6)
    ax.imshow(img_n, cmap="gray")
    
    if (gt == lv_label).any():
        ax.contour(gt == lv_label, levels=[0.5], colors="red", linewidths=1.5)
    if (pr == lv_label).any():
        ax.contour(pr == lv_label, levels=[0.5], colors="lime", linewidths=1.5, linestyles="--")

def _error_map(gt, pr):
    """
    Helper to generate TP/FP/FN RGB map.
    """
    H, W = gt.shape
    canvas = np.zeros((H, W, 3), dtype=float)
    tp = (gt > 0) & (pr > 0)
    fp = (gt == 0) & (pr > 0)
    fn = (gt > 0) & (pr == 0)
    canvas[tp] = (1.0, 1.0, 1.0)      # white
    canvas[fp] = (1.0, 0.55, 0.0)     # orange
    canvas[fn] = (0.1, 0.4, 1.0)      # blue
    return canvas

def plot_qualitative_grid(samples, save_path=None):
    """
    Plots specific sample cases (overlay, error map, labels).
    
    Args:
        samples (list): List of sample dictionaries.
        save_path (str): Optional save path.
        
    Returns:
        matplotlib.figure.Figure: Generated figure.
    """
    setup_style()
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 3, figsize=(7.0, 2.5 * n_rows), dpi=300)
    if n_rows == 1: axes = np.expand_dims(axes, axis=0)

    for r, s in enumerate(samples):
        img, gt, pr = s['img'], s['gt'], s['pr']
        
        # Overlay
        _overlay_contours(axes[r, 0], img, gt, pr)
        axes[r, 0].set_title(s.get('title', 'Overlay'))
        
        # Error Map
        axes[r, 1].imshow(_error_map(gt, pr))
        axes[r, 1].set_title("Error Map")
        
        # GT Labels
        axes[r, 2].imshow(color.label2rgb(gt, image=img, bg_label=0, alpha=0.4))
        axes[r, 2].set_title("GT Labels")

        for ax in axes[r]: ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig

def plot_ef_category_roc(df_ef, save_path=None):
    """
    Plots Confusion Matrix and ROC for Low EF detection.
    
    Args:
        df_ef (pd.DataFrame): DataFrame containing EF metrics.
        save_path (str): Optional save path.
        
    Returns:
        matplotlib.figure.Figure: Generated figure.
    """
    setup_style()
    bins = [0.0, 45.0, 55.0, 100.0]
    labels = ["EF < 45%", "45-55%", "EF > 55%"]
    
    df_ef["cat_ref"] = pd.cut(df_ef["EF_ref"], bins=bins, labels=labels)
    df_ef["cat_pred"] = pd.cut(df_ef["EF_pred"], bins=bins, labels=labels)
    
    # Confusion Matrix
    cm = pd.crosstab(df_ef["cat_ref"], df_ef["cat_pred"])
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    
    # ROC for Low EF (<45%)
    y_true = (df_ef["EF_ref"] < 45.0).astype(int)
    scores = 45.0 - df_ef["EF_pred"] 
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    
    # Plot CM
    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_title("Confusion Matrix")
    ax1.set_xticks(range(3)); ax1.set_xticklabels(labels, rotation=15)
    ax1.set_yticks(range(3)); ax1.set_yticklabels(labels)
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, cm.iloc[i, j], ha="center", va="center", 
                     color="white" if cm.iloc[i, j] > cm.max().max()/2 else "black")

    # Plot ROC
    ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax2.plot([0,1], [0,1], 'k--')
    ax2.set_title("Low EF (<45%) ROC")
    ax2.legend()
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    return fig

def plot_results_table(df_data, title, save_path=None):
    """
    Renders a DataFrame as a figure table.
    
    Args:
        df_data (pd.DataFrame): Data to display.
        title (str): Table title.
        save_path (str): Optional save path.
        
    Returns:
        matplotlib.figure.Figure: Generated figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, len(df_data)*0.5 + 1), dpi=300)
    ax.axis("off")
    
    tbl = ax.table(
        cellText=df_data.values,
        colLabels=df_data.columns,
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.5)
    
    ax.set_title(title, fontweight="bold")
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_coverage_by_difficulty(df_bins, save_path=None):
    """
    Plots coverage comparison between Baseline and MACS across difficulty bins.
    
    Args:
        df_bins (pd.DataFrame): DataFrame with index ["Easy", "Medium", "Hard"] 
                                and columns ["Baseline", "MACS"].
        save_path (str): Optional path to save the figure.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    setup_style()
    
    # Re-order if needed
    desired_order = ["Easy", "Medium", "Hard"]
    df = df_bins.reindex(desired_order)
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    # Bar settings
    x = np.arange(len(df.index))
    width = 0.35
    
    # Colors: Baseline (Grey/Blue), MACS (Green/Highlight)
    # Using a professional palette
    color_base = "#7f7f7f"  # Grey
    color_macs = "#2ca02c"  # Green
    
    rects1 = ax.bar(x - width/2, df["Baseline"], width, label='Baseline', color=color_base, alpha=0.8, edgecolor='k')
    rects2 = ax.bar(x + width/2, df["MACS"], width, label='MACS (Yours)', color=color_macs, alpha=0.9, edgecolor='k')
    
    # Target Line
    ax.axhline(y=0.90, color='r', linestyle='--', linewidth=1.2, label='Target (90%)')
    
    # Labels
    ax.set_ylabel('Coverage (Pass Rate)')
    ax.set_title('Coverage by Difficulty')
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.legend(loc='lower left')
    
    ax.set_ylim(0.0, 1.05)
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    return fig
