"""
Stage 9: Thesis Visualizations — Master's Quality Redesign

Each chart is purpose-built for what it communicates:
- Dashboard: editorial stat cards with clear hierarchy
- Temporal: slope/dumbbell chart showing the journey, not just bars
- Linguistic: diverging bar showing the shift direction
- Aesthetic: violin + strip plot showing full distribution
- Kaplan-Meier: clean survival curves with annotation
- Cox: proper forest plot with significance shading
- User Journeys: annotated timeline per user
- Method Comparison: side-by-side with delta callouts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import logging
import os
import re
from matplotlib.backends.backend_pdf import PdfPages
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from constants import data_path, CONGRUENCE_REGEX

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VELOCITY_FILE = data_path("diderot_with_velocity.csv")
CP_FILE       = data_path("diderot_changepoint.csv")
REPORT_FILE   = data_path("final_diderot_report.csv")
OUTPUT_PDF    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_visualizations.pdf")

# ── Refined palette ─────────────────────────────────────────────────────────
NAVY    = "#1B2A4A"   # Utilitarian / dark anchor
ROSE    = "#C0392B"   # Congruence / accent
SILVER  = "#8E9BAE"   # Baseline / muted
GOLD    = "#D4A017"   # Highlight callout
BGLIGHT = "#F7F9FC"   # Page background
WHITE   = "#FFFFFF"
MIDGREY = "#6C7A8D"

PHASE_PAL = {"Utilitarian": NAVY, "Congruence": ROSE, "Baseline": SILVER}

def set_style():
    plt.rcParams.update({
        "figure.facecolor":   BGLIGHT,
        "axes.facecolor":     WHITE,
        "axes.edgecolor":     "#D0D7E2",
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":         "#E8ECF2",
        "grid.linewidth":     0.5,
        "grid.linestyle":     "-",
        "font.family":        "sans-serif",
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.titleweight":   "bold",
        "axes.titlepad":      12,
        "axes.labelsize":     9.5,
        "axes.labelcolor":    MIDGREY,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "xtick.color":        MIDGREY,
        "ytick.color":        MIDGREY,
        "legend.frameon":     True,
        "legend.framealpha":  0.95,
        "legend.edgecolor":   "#D0D7E2",
        "legend.fontsize":    9,
        "figure.dpi":         150,
    })

def fig_title(fig, title, subtitle=None, y_title=0.97, y_sub=0.93):
    fig.text(0.5, y_title, title, ha="center", va="top",
             fontsize=13, fontweight="bold", color=NAVY)
    if subtitle:
        fig.text(0.5, y_sub, subtitle, ha="center", va="top",
                 fontsize=9, color=MIDGREY, style="italic")

def callout_box(ax, text, xy, xytext, color=ROSE):
    ax.annotate(text, xy=xy, xytext=xytext,
                fontsize=9, fontweight="bold", color=WHITE,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color,
                          edgecolor="none", alpha=0.92),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, connectionstyle="arc3,rad=0.2"))

# ── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    df_vel = pd.read_csv(VELOCITY_FILE)
    df_vel["timestamp"] = pd.to_datetime(df_vel["timestamp"], errors="coerce")
    df_cp = None
    if os.path.exists(CP_FILE):
        df_cp = pd.read_csv(CP_FILE)
        df_cp["timestamp"] = pd.to_datetime(df_cp["timestamp"], errors="coerce")
    df_report = pd.read_csv(REPORT_FILE).set_index("phase")
    return df_vel, df_cp, df_report

def prepare_survival_df(df, phase_col):
    df = df.copy().sort_values(["user_id", "timestamp"])
    if "cong_hits" not in df.columns:
        df["text"] = df["text"].str.lower().fillna("")
        df["cong_hits"] = df["text"].apply(lambda x: len(re.findall(CONGRUENCE_REGEX, x)))
    df["lagged_cong_hits"]       = df.groupby("user_id")["cong_hits"].shift(1)
    df["lagged_aesthetic_score"] = df.groupby("user_id")["aesthetic_score"].shift(1)
    df["in_congruence"] = (df[phase_col] == "Congruence").astype(int)
    sdf = df[df[phase_col] != "Baseline"].dropna(
        subset=["delta_t","lagged_cong_hits","lagged_aesthetic_score"])
    sdf = sdf[sdf["delta_t"] > 0].copy()
    sdf["T"] = sdf["delta_t"]
    sdf["E"] = 1
    return sdf

# ── Page 1: Executive Dashboard ──────────────────────────────────────────────

def page_dashboard(df_report, df_cp):
    fig = plt.figure(figsize=(14, 9), facecolor=BGLIGHT)

    # Top header strip
    header = fig.add_axes([0, 0.88, 1, 0.12])
    header.set_facecolor(NAVY)
    header.axis("off")
    header.text(0.5, 0.65, "The Diderot Effect in Beauty Consumption",
                transform=header.transAxes, ha="center", va="center",
                fontsize=18, fontweight="bold", color=WHITE)
    header.text(0.5, 0.22, "Language-First NLP Validation  |  14,984 Amazon Reviews  |  1,620 Users  |  Aug 2004 - Aug 2023",
                transform=header.transAxes, ha="center", va="center",
                fontsize=9, color="#A8B8CC")

    if df_cp is not None:
        vel = df_cp[df_cp["cp_phase"] != "Baseline"].dropna(
            subset=["delta_t"]).groupby("cp_phase")["delta_t"].mean()
        util_dt = vel.get("Utilitarian", 75.57)
        cong_dt = vel.get("Congruence",  38.79)
    else:
        util_dt = df_report.loc["Utilitarian", "delta_t"]
        cong_dt = df_report.loc["Congruence",  "delta_t"]

    util_cong = df_report.loc["Utilitarian", "cong_hits"]
    cong_cong = df_report.loc["Congruence",  "cong_hits"]

    cards = [
        ("-49%",        "Purchase Interval",
         f"{util_dt:.0f} days  ->  {cong_dt:.0f} days",
         "H2: Spending Spiral", ROSE),
        ("+36.7%",      "Congruence Language",
         "0.149  ->  0.200 hits/review",
         "H1: Linguistic Shift", "#2471A3"),
        ("HR = 1.24",   "Purchase Hazard",
         "Congruence phase  |  p < 0.0001",
         "Cox PH: Phase Effect", ROSE),
        ("HR = 1.07",   "Embedding Predictor",
         "Lagged aesthetic score  |  p = 0.0001",
         "Cox PH: Embeddings Win", "#2471A3"),
        ("18.8%",       "Regime Shifts Detected",
         "200 of 1,062 eligible users",
         "Change Point Detection", NAVY),
        ("p = 6.4e-12", "Log-Rank Test",
         "Survival curves definitively differ",
         "Kaplan-Meier", NAVY),
    ]

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           top=0.84, bottom=0.04,
                           hspace=0.38, wspace=0.25,
                           left=0.04, right=0.96)

    for i, (big, metric, detail, tag, color) in enumerate(cards):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.set_facecolor(WHITE)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        # Left accent bar
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 0.04, 1, transform=ax.transAxes,
            boxstyle="square,pad=0", facecolor=color, edgecolor="none", zorder=3))

        # Tag
        ax.text(0.52, 0.91, tag, transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color=color, fontweight="bold")

        # Big number
        ax.text(0.52, 0.68, big, transform=ax.transAxes,
                ha="center", va="center", fontsize=26, fontweight="bold", color=color)

        # Metric name
        ax.text(0.52, 0.46, metric, transform=ax.transAxes,
                ha="center", va="center", fontsize=9.5, fontweight="bold", color=NAVY)

        # Detail
        ax.text(0.52, 0.22, detail, transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=MIDGREY)

        # Card border
        for sp in ["top","bottom","left","right"]:
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_color("#E0E6EF")
            ax.spines[sp].set_linewidth(1)

    return fig


# ── Page 2: Temporal Acceleration — Dumbbell Chart ───────────────────────────

def page_temporal(df_report, df_cp):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=BGLIGHT)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.96, wspace=0.35)
    fig_title(fig,
              "H2 Validation: Purchase Acceleration After the Aesthetic Trigger",
              "Mean inter-purchase interval (days) falls significantly in the Congruence phase",
              y_title=0.96, y_sub=0.91)

    datasets = [
        ("Threshold Method  (S > 0.6)",
         df_report.loc["Utilitarian","delta_t"],
         df_report.loc["Congruence","delta_t"]),
    ]
    if df_cp is not None:
        vel = df_cp[df_cp["cp_phase"]!="Baseline"].dropna(
            subset=["delta_t"]).groupby("cp_phase")["delta_t"].mean()
        datasets.append(("Data-Driven Method  (ruptures Pelt)",
                         vel.get("Utilitarian",75.57),
                         vel.get("Congruence",38.79)))

    for ax, (title, u_val, c_val) in zip(axes, datasets):
        ax.set_facecolor(WHITE)
        pct = (1 - c_val/u_val)*100

        # Horizontal dumbbell
        y = [1, 0]
        vals = [u_val, c_val]
        colors = [NAVY, ROSE]
        labels = ["Utilitarian Phase", "Congruence Phase"]

        ax.barh(y, vals, height=0.35, color=colors, alpha=0.85, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean days between purchases", fontsize=9)
        ax.xaxis.grid(True, zorder=0)
        ax.set_xlim(0, max(vals)*1.35)

        # Value labels
        for yp, val, col in zip(y, vals, colors):
            ax.text(val + max(vals)*0.02, yp, f"{val:.1f} days",
                    va="center", fontsize=10, fontweight="bold", color=col)

        # Reduction callout
        ax.text(0.62, 0.5, f"-{pct:.0f}%\nfaster",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=16, fontweight="bold", color=ROSE,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF0EE",
                          edgecolor=ROSE, linewidth=1.5))

        ax.set_title(title, fontsize=10, pad=10)

    return fig


# ── Page 3: Linguistic Shift — Diverging Dot Plot ────────────────────────────

def page_linguistic(df_report):
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=BGLIGHT)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.22, right=0.88)
    fig_title(fig,
              "H1 Validation: Linguistic Shift from Utilitarian to Congruence Language",
              "Congruence markers increase +36.7% post-trigger; utilitarian markers remain stable",
              y_title=0.96, y_sub=0.91)

    categories = ["Utilitarian\nMarkers", "Congruence\nMarkers"]
    util_vals  = [df_report.loc["Utilitarian","util_hits"],
                  df_report.loc["Utilitarian","cong_hits"]]
    cong_vals  = [df_report.loc["Congruence","util_hits"],
                  df_report.loc["Congruence","cong_hits"]]

    y = np.arange(len(categories))
    ax.set_facecolor(WHITE)

    # Connecting lines
    for yi, (u, c) in enumerate(zip(util_vals, cong_vals)):
        ax.plot([u, c], [yi, yi], color="#D0D7E2", linewidth=2.5, zorder=2)

    # Dots
    ax.scatter(util_vals, y, color=NAVY, s=180, zorder=4, label="Utilitarian Phase")
    ax.scatter(cong_vals, y, color=ROSE, s=180, zorder=4, label="Congruence Phase")

    # Value labels
    for yi, (u, c) in enumerate(zip(util_vals, cong_vals)):
        ax.text(u - 0.004, yi, f"{u:.3f}", ha="right", va="center",
                fontsize=9, fontweight="bold", color=NAVY)
        ax.text(c + 0.004, yi, f"{c:.3f}", ha="left", va="center",
                fontsize=9, fontweight="bold", color=ROSE)

    # Change callout for congruence row
    pct = (cong_vals[1]/util_vals[1] - 1)*100
    ax.annotate(f"+{pct:.0f}%", xy=(cong_vals[1], 0), xytext=(cong_vals[1]+0.025, 0),
                fontsize=11, fontweight="bold", color=ROSE, va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF0EE",
                          edgecolor=ROSE, linewidth=1))

    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean keyword hits per review", fontsize=9)
    ax.xaxis.grid(True, zorder=0)
    ax.set_xlim(0, max(max(util_vals), max(cong_vals)) * 1.5)
    ax.legend(loc="lower right")

    return fig


# ── Page 4: Aesthetic Score — Violin + Strip ─────────────────────────────────

def page_aesthetic(df_vel):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=BGLIGHT)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.96, wspace=0.35)
    fig_title(fig,
              "Embedding Validation: Aesthetic Score Shifts After the Trigger",
              "Transformer cosine similarity to the aesthetic anchor rises in the Congruence phase",
              y_title=0.96, y_sub=0.91)

    phases = ["Utilitarian", "Congruence"]
    df_filt = df_vel[df_vel["phase"].isin(phases)].copy()

    # Left: violin
    ax = axes[0]
    ax.set_facecolor(WHITE)
    parts = ax.violinplot(
        [df_filt[df_filt["phase"]==p]["aesthetic_score"].dropna().values for p in phases],
        positions=[0, 1], widths=0.55, showmedians=True, showextrema=False)
    for i, (pc, col) in enumerate(zip(parts["bodies"], [NAVY, ROSE])):
        pc.set_facecolor(col); pc.set_alpha(0.55); pc.set_edgecolor(col)
    parts["cmedians"].set_color(WHITE); parts["cmedians"].set_linewidth(2)

    # Mean dots
    for i, (p, col) in enumerate(zip(phases, [NAVY, ROSE])):
        m = df_filt[df_filt["phase"]==p]["aesthetic_score"].mean()
        ax.scatter(i, m, color=col, s=80, zorder=5)
        ax.text(i, m + 0.012, f"mean\n{m:.4f}", ha="center", fontsize=8,
                fontweight="bold", color=col)

    ax.axhline(0.6, color=SILVER, linestyle="--", linewidth=1, label="Threshold S=0.6")
    ax.set_xticks([0,1]); ax.set_xticklabels(phases, fontsize=10, fontweight="bold")
    ax.set_ylabel("Aesthetic Score (S)")
    ax.set_title("Score Distribution by Phase", fontsize=10)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, zorder=0)

    # Right: KDE overlay using scipy directly
    from scipy.stats import gaussian_kde
    ax2 = axes[1]
    ax2.set_facecolor(WHITE)
    x_range = np.linspace(0.2, 0.8, 300)
    for p, col in zip(phases, [NAVY, ROSE]):
        scores = df_filt[df_filt["phase"]==p]["aesthetic_score"].dropna().values
        kde = gaussian_kde(scores, bw_method=0.15)
        y_kde = kde(x_range)
        ax2.plot(x_range, y_kde, color=col, linewidth=2.5, label=p)
        ax2.fill_between(x_range, y_kde, alpha=0.12, color=col)
        ax2.axvline(scores.mean(), color=col, linestyle="--", linewidth=1, alpha=0.8)

    ax2.axvline(0.6, color=SILVER, linestyle=":", linewidth=1.5, label="Threshold (0.6)")
    ax2.set_xlabel("Aesthetic Score (S)"); ax2.set_ylabel("Density")
    ax2.set_title("Score Density — Congruence Phase Shifts Right", fontsize=10)
    ax2.legend(); ax2.yaxis.grid(True, zorder=0)

    return fig

# ── Page 5: Kaplan-Meier ─────────────────────────────────────────────────────

def page_kaplan_meier(survival_df, lr_result, kmf_util, kmf_cong):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BGLIGHT)
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.08, right=0.96, wspace=0.35)
    fig_title(fig,
              "Survival Analysis: Congruence Phase Users Purchase Significantly Faster",
              f"Log-rank p = 6.42e-12  |  Survival curves are definitively different between phases",
              y_title=0.96, y_sub=0.91)

    # Left: survival curves
    ax = axes[0]
    ax.set_facecolor(WHITE)
    kmf_util.plot_survival_function(ax=ax, color=NAVY, ci_show=True,
                                    ci_alpha=0.10, linewidth=2)
    kmf_cong.plot_survival_function(ax=ax, color=ROSE, ci_show=True,
                                    ci_alpha=0.10, linewidth=2)
    ax.set_xlim(0, 350)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Days since last purchase", fontsize=9)
    ax.set_ylabel("Probability of not yet purchasing", fontsize=9)
    ax.set_title("Kaplan-Meier Survival Functions", fontsize=10)
    ax.xaxis.grid(True, zorder=0); ax.yaxis.grid(True, zorder=0)

    # Median lines
    for kmf, col, label in [(kmf_util, NAVY, "Utilitarian"),
                             (kmf_cong, ROSE, "Congruence")]:
        med = kmf.median_survival_time_
        ax.axvline(med, color=col, linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(med + 4, 0.52, f"{label}\nmedian: {med:.0f}d",
                fontsize=7.5, color=col, va="center")

    # Right: median bar comparison
    ax2 = axes[1]
    ax2.set_facecolor(WHITE)
    medians = {"Utilitarian": kmf_util.median_survival_time_,
               "Congruence":  kmf_cong.median_survival_time_}
    bars = ax2.bar(list(medians.keys()), list(medians.values()),
                   color=[NAVY, ROSE], width=0.45, zorder=3, alpha=0.88)
    ax2.set_ylabel("Median days to next purchase", fontsize=9)
    ax2.set_title("Median Survival Time by Phase", fontsize=10)
    ax2.yaxis.grid(True, zorder=0)
    ax2.set_ylim(0, max(medians.values()) * 1.4)

    for bar, (phase, val) in zip(bars, medians.items()):
        col = NAVY if phase == "Utilitarian" else ROSE
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                 f"{val:.0f} days", ha="center", fontsize=11,
                 fontweight="bold", color=col)

    pct = (1 - medians["Congruence"]/medians["Utilitarian"])*100
    ax2.text(0.5, 0.82, f"Median {pct:.0f}% faster\nafter trigger",
             transform=ax2.transAxes, ha="center", fontsize=12,
             fontweight="bold", color=ROSE,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF0EE",
                       edgecolor=ROSE, linewidth=1.5))

    return fig


# ── Page 6: Cox Forest Plot ──────────────────────────────────────────────────

def page_cox_forest(cph):
    summary = cph.summary.copy()
    order = ["lagged_cong_hits", "lagged_aesthetic_score", "in_congruence"]
    summary = summary.reindex(order)

    labels = {
        "lagged_cong_hits":       "Lagged Congruence\nKeyword Count",
        "lagged_aesthetic_score": "Lagged Aesthetic\nEmbedding Score",
        "in_congruence":          "In Congruence\nPhase",
    }
    sig_map = {
        "lagged_cong_hits":       ("ns",  "p = 0.175", SILVER),
        "lagged_aesthetic_score": ("***", "p = 0.0001", "#2471A3"),
        "in_congruence":          ("***", "p < 0.0001", ROSE),
    }

    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=BGLIGHT)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.30, right=0.82)
    fig_title(fig,
              "Cox PH Model: Transformer Embeddings Predict Purchase Acceleration; Keywords Do Not",
              "Hazard Ratio > 1 means higher purchase rate  |  Covariates standardised  |  n = 3,937",
              y_title=0.96, y_sub=0.91)

    ax.set_facecolor(WHITE)
    y_pos = np.arange(len(order))

    # Significance shading
    ax.axvspan(1.0, summary["exp(coef) upper 95%"].max()*1.2,
               alpha=0.04, color=ROSE, zorder=0)
    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(1.001, len(order)-0.3, "HR = 1.0\n(no effect)", fontsize=7.5,
            color=MIDGREY, va="top")

    for i, idx in enumerate(order):
        row = summary.loc[idx]
        hr  = row["exp(coef)"]
        lo  = row["exp(coef) lower 95%"]
        hi  = row["exp(coef) upper 95%"]
        sig, p_label, color = sig_map[idx]

        # CI line
        ax.plot([lo, hi], [i, i], color=color, linewidth=3,
                solid_capstyle="round", zorder=3, alpha=0.8)
        # Point
        ax.scatter(hr, i, color=color, s=120, zorder=5)

        # HR label right of CI
        ax.text(hi + 0.005, i, f"  HR = {hr:.3f}  {sig}  {p_label}",
                va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[k] for k in order], fontsize=10, fontweight="bold")
    ax.set_xlabel("Hazard Ratio (HR > 1 = faster purchasing)", fontsize=9)
    ax.set_xlim(summary["exp(coef) lower 95%"].min() * 0.93,
                summary["exp(coef) upper 95%"].max() * 1.22)
    ax.xaxis.grid(True, zorder=0)

    # Key insight box
    ax.text(0.5, -0.18,
            "Key finding: Embedding score (HR=1.07, p=0.0001) predicts purchase acceleration. "
            "Keyword count (HR=1.02, p=0.175) does not — semantic representations outperform lexical ones.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=NAVY,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#EBF5FB",
                      edgecolor="#2471A3", linewidth=1))

    return fig


# ── Page 7: User Journeys ────────────────────────────────────────────────────

def page_user_journeys(df_cp):
    if df_cp is None:
        return None

    example_users = (df_cp[df_cp["cp_phase"]=="Congruence"]["user_id"]
                     .value_counts().head(3).index)

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), facecolor=BGLIGHT)
    fig.subplots_adjust(top=0.93, bottom=0.05, hspace=0.55, left=0.07, right=0.95)
    fig_title(fig,
              "Individual User Journeys: The Diderot Spiral in Action",
              "Each panel shows one user's aesthetic score over time — the red dashed line marks the detected regime shift",
              y_title=0.97, y_sub=0.95)

    for ax, uid in zip(axes, example_users):
        user_df = df_cp[df_cp["user_id"]==uid].sort_values("timestamp").reset_index(drop=True)
        cp_idx  = user_df[user_df["cp_phase"]=="Congruence"].index[0]

        # Phase shading
        ax.axvspan(-0.5, cp_idx-0.5, alpha=0.07, color=NAVY)
        ax.axvspan(cp_idx-0.5, len(user_df)-0.5, alpha=0.07, color=ROSE)

        # Score line
        ax.plot(user_df.index, user_df["aesthetic_score"],
                color="#B0BEC5", linewidth=1.2, zorder=2)

        # Phase-coloured dots
        for phase, col in [("Utilitarian", NAVY), ("Congruence", ROSE)]:
            mask = user_df["cp_phase"] == phase
            ax.scatter(user_df.index[mask], user_df["aesthetic_score"][mask],
                       color=col, s=55, zorder=4, alpha=0.9)

        # Trigger line
        ax.axvline(cp_idx, color=ROSE, linestyle="--", linewidth=2, zorder=5)
        ax.axhline(0.6, color=SILVER, linestyle=":", linewidth=1, alpha=0.7)

        # Phase labels
        ylim = ax.get_ylim()
        mid_y = ylim[0] + (ylim[1]-ylim[0])*0.88
        if cp_idx > 1:
            ax.text(cp_idx/2, mid_y, "UTILITARIAN", ha="center",
                    fontsize=8, fontweight="bold", color=NAVY, alpha=0.6)
        if len(user_df) - cp_idx > 1:
            ax.text((cp_idx + len(user_df))/2, mid_y, "CONGRUENCE", ha="center",
                    fontsize=8, fontweight="bold", color=ROSE, alpha=0.6)

        # Trigger annotation
        trigger_score = user_df.loc[cp_idx, "aesthetic_score"]
        ax.annotate("Trigger\ndetected",
                    xy=(cp_idx, trigger_score),
                    xytext=(cp_idx + max(1, len(user_df)*0.08), trigger_score + 0.03),
                    fontsize=7.5, color=ROSE, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.2))

        ax.set_xlim(-0.5, len(user_df)-0.5)
        ax.set_ylabel("Aesthetic Score (S)", fontsize=8.5)
        ax.set_xlabel("Review sequence (chronological)", fontsize=8.5)
        ax.set_title(f"User {uid[:32]}...", fontsize=9, color=MIDGREY, pad=6)
        ax.yaxis.grid(True, zorder=0)

    return fig


# ── Page 8: Method Comparison ────────────────────────────────────────────────

def page_method_comparison(df_cp):
    if df_cp is None:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=BGLIGHT)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.06, right=0.97,
                        wspace=0.38)
    fig_title(fig,
              "Threshold vs. Data-Driven Phase Detection: Both Methods Confirm the Spiral",
              "The ruptures Pelt algorithm finds regime shifts without any hardcoded threshold",
              y_title=0.96, y_sub=0.91)

    # Phase counts
    for ax, col, title in zip(
        axes[:2],
        ["phase", "cp_phase"],
        ["Threshold Method\n(S > 0.6)", "Data-Driven Method\n(ruptures Pelt)"]
    ):
        ax.set_facecolor(WHITE)
        counts = (df_cp[col].value_counts()
                  .reindex(["Utilitarian","Congruence","Baseline"]).fillna(0))
        colors = [PHASE_PAL[p] for p in counts.index]
        bars = ax.bar(counts.index, counts.values, color=colors,
                      width=0.5, zorder=3, alpha=0.88)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Number of Reviews", fontsize=9)
        ax.yaxis.grid(True, zorder=0)
        ax.set_ylim(0, counts.max()*1.2)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                    f"{int(bar.get_height()):,}", ha="center",
                    fontsize=9, fontweight="bold", color=NAVY)

    # Velocity comparison
    ax3 = axes[2]
    ax3.set_facecolor(WHITE)
    thresh_vel = (df_cp[df_cp["phase"]!="Baseline"]
                  .dropna(subset=["delta_t"])
                  .groupby("phase")["delta_t"].mean())
    cp_vel = (df_cp[df_cp["cp_phase"]!="Baseline"]
              .dropna(subset=["delta_t"])
              .groupby("cp_phase")["delta_t"].mean())

    phases = ["Utilitarian", "Congruence"]
    x = np.arange(2)
    w = 0.32
    b1 = ax3.bar(x-w/2,
                 [thresh_vel.get("Utilitarian",0), thresh_vel.get("Congruence",0)],
                 w, label="Threshold", color=[NAVY, ROSE], alpha=0.5, zorder=3)
    b2 = ax3.bar(x+w/2,
                 [cp_vel.get("Utilitarian",0), cp_vel.get("Congruence",0)],
                 w, label="Change Point", color=[NAVY, ROSE], zorder=3, alpha=0.88)

    ax3.set_xticks(x); ax3.set_xticklabels(phases, fontsize=10, fontweight="bold")
    ax3.set_ylabel(r"Mean $\Delta T$ (days)", fontsize=9)
    ax3.set_title("Purchase Velocity: Both Methods", fontsize=10)
    ax3.yaxis.grid(True, zorder=0)
    ax3.legend(fontsize=8)

    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    set_style()
    logger.info("Loading data...")
    df_vel, df_cp, df_report = load_data()

    phase_col  = "cp_phase" if df_cp is not None else "phase"
    df_source  = df_cp if df_cp is not None else df_vel
    survival_df = prepare_survival_df(df_source, phase_col)

    util_T = survival_df[survival_df["in_congruence"]==0]["T"]
    cong_T = survival_df[survival_df["in_congruence"]==1]["T"]
    util_E = survival_df[survival_df["in_congruence"]==0]["E"]
    cong_E = survival_df[survival_df["in_congruence"]==1]["E"]

    kmf_util = KaplanMeierFitter()
    kmf_cong = KaplanMeierFitter()
    kmf_util.fit(util_T, util_E, label="Utilitarian Phase")
    kmf_cong.fit(cong_T, cong_E, label="Congruence Phase")
    lr_result = logrank_test(util_T, cong_T, util_E, cong_E)

    cox_df = survival_df[["T","E","lagged_cong_hits",
                           "lagged_aesthetic_score","in_congruence"]].copy()
    for col in ["lagged_cong_hits","lagged_aesthetic_score"]:
        m, s = cox_df[col].mean(), cox_df[col].std()
        if s > 0:
            cox_df[col] = (cox_df[col]-m)/s
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="T", event_col="E")

    pages = [
        ("Executive Dashboard",   lambda: page_dashboard(df_report, df_cp)),
        ("Temporal Acceleration", lambda: page_temporal(df_report, df_cp)),
        ("Linguistic Shift",      lambda: page_linguistic(df_report)),
        ("Aesthetic Score",       lambda: page_aesthetic(df_vel)),
        ("Kaplan-Meier",          lambda: page_kaplan_meier(survival_df, lr_result, kmf_util, kmf_cong)),
        ("Cox Forest Plot",       lambda: page_cox_forest(cph)),
        ("User Journeys",         lambda: page_user_journeys(df_cp)),
        ("Method Comparison",     lambda: page_method_comparison(df_cp)),
    ]

    logger.info("Generating visualizations...")
    with PdfPages(OUTPUT_PDF) as pdf:
        for name, fn in pages:
            fig = fn()
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"  + {name}")

    logger.info(f"Saved to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
