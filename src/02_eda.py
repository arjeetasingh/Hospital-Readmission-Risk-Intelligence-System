"""
Hospital Readmission Risk Intelligence System
Step 02 — Exploratory Data Analysis (EDA)

Generates publication-quality plots saved to outputs/eda/
Run after 01_load_and_clean.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DB_USER  = os.getenv("DB_USER", "root")
DB_PASS  = os.getenv("DB_PASS", "")
DB_HOST  = os.getenv("DB_HOST", "localhost")
DB_PORT  = os.getenv("DB_PORT", "3306")
DB_NAME  = os.getenv("DB_NAME", "hospital_readmission")

OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {
    "primary":   "#2C5F8A",
    "secondary": "#E05C2A",
    "neutral":   "#6B6B6B",
    "light":     "#D6E8F5",
    "green":     "#2A7A45",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})


def get_df() -> pd.DataFrame:
    engine = create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return pd.read_sql("SELECT * FROM patient_features", engine)


# ── Individual plots ──────────────────────────────────────────────────────────

def plot_readmission_distribution(df: pd.DataFrame):
    counts = pd.Series({
        "Readmitted < 30 days": df["readmitted_30"].sum(),
        "Readmitted > 30 days": (df["readmitted_any"] - df["readmitted_30"]).sum(),
        "Not Readmitted":       (~df["readmitted_any"].astype(bool)).sum(),
    })
    pct = (counts / counts.sum() * 100).round(1)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        counts.index, counts.values,
        color=[PALETTE["secondary"], PALETTE["primary"], PALETTE["green"]],
        edgecolor="white", linewidth=1.5, width=0.55
    )
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200,
                f"{p}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Readmission Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Patient Encounters")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_readmission_distribution.png"))
    plt.close()
    print("[EDA] Saved: 01_readmission_distribution.png")


def plot_age_vs_readmission(df: pd.DataFrame):
    bins   = [0, 30, 50, 70, 110]
    labels = ["0-30", "31-50", "51-70", "71+"]
    df["age_group"] = pd.cut(df["age_mid"], bins=bins, labels=labels, right=True)

    agg = df.groupby("age_group", observed=True).agg(
        total      = ("readmitted_30", "count"),
        readmitted = ("readmitted_30", "sum")
    ).reset_index()
    agg["rate"] = agg["readmitted"] / agg["total"] * 100

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.bar(agg["age_group"], agg["total"],
            color=PALETTE["light"], edgecolor=PALETTE["primary"],
            linewidth=1.2, label="Total Encounters", zorder=2)
    ax2.plot(agg["age_group"], agg["rate"],
             color=PALETTE["secondary"], marker="o",
             linewidth=2.5, markersize=8, label="30-day Readmission Rate (%)", zorder=3)

    ax1.set_xlabel("Age Group")
    ax1.set_ylabel("Total Encounters", color=PALETTE["primary"])
    ax2.set_ylabel("30-day Readmission Rate (%)", color=PALETTE["secondary"])
    ax2.set_ylim(0, 20)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax1.set_title("Age Group vs Readmission Rate", fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_age_vs_readmission.png"))
    plt.close()
    print("[EDA] Saved: 02_age_vs_readmission.png")


def plot_los_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (label, sub) in zip(axes, [("Not Readmitted <30d", df[df["readmitted_30"]==0]),
                                        ("Readmitted <30d",     df[df["readmitted_30"]==1])]):
        color = PALETTE["primary"] if label.startswith("Not") else PALETTE["secondary"]
        ax.hist(sub["time_in_hospital"], bins=range(1, 15),
                color=color, edgecolor="white", linewidth=0.8, density=True)
        ax.axvline(sub["time_in_hospital"].mean(), color="black",
                   linestyle="--", linewidth=1.5,
                   label=f'Mean = {sub["time_in_hospital"].mean():.1f}d')
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Length of Stay (days)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle("Length of Stay Distribution by Readmission Status",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_los_distribution.png"))
    plt.close()
    print("[EDA] Saved: 03_los_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    num_cols = [
        "age_mid", "time_in_hospital", "num_lab_procedures",
        "num_procedures", "num_medications", "total_visits",
        "number_diagnoses", "readmitted_30"
    ]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="coolwarm", center=0,
        annot=True, fmt=".2f", annot_kws={"size": 9},
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_correlation_heatmap.png"))
    plt.close()
    print("[EDA] Saved: 04_correlation_heatmap.png")


def plot_insulin_effect(df: pd.DataFrame):
    agg = df.groupby("on_insulin").agg(
        rate=("readmitted_30", "mean"),
        total=("readmitted_30", "count")
    ).reset_index()
    agg["label"] = agg["on_insulin"].map({0: "Not on Insulin", 1: "On Insulin"})
    agg["rate"] *= 100

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(agg["label"], agg["rate"],
                  color=[PALETTE["primary"], PALETTE["secondary"]],
                  width=0.45, edgecolor="white", linewidth=1.5)
    for bar, (_, row) in zip(bars, agg.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{row["rate"]:.1f}%\n(n={row["total"]:,})',
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("30-day Readmission Rate (%)")
    ax.set_title("Insulin Prescription & 30-day Readmission Rate",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0, 20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_insulin_effect.png"))
    plt.close()
    print("[EDA] Saved: 05_insulin_effect.png")


def plot_medications_boxplot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [
        df.loc[df["readmitted_30"]==0, "num_medications"],
        df.loc[df["readmitted_30"]==1, "num_medications"],
    ]
    bp = ax.boxplot(groups, patch_artist=True, notch=True,
                    medianprops=dict(color="white", linewidth=2.5),
                    widths=0.4)
    colors = [PALETTE["primary"], PALETTE["secondary"]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticklabels(["Not Readmitted <30d", "Readmitted <30d"])
    ax.set_ylabel("Number of Medications")
    ax.set_title("Medication Count by Readmission Status",
                 fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_medications_boxplot.png"))
    plt.close()
    print("[EDA] Saved: 06_medications_boxplot.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[INFO] Loading data from MySQL...")
    df = get_df()
    print(f"[INFO] Loaded {len(df):,} rows.")

    plot_readmission_distribution(df)
    plot_age_vs_readmission(df)
    plot_los_distribution(df)
    plot_correlation_heatmap(df)
    plot_insulin_effect(df)
    plot_medications_boxplot(df)

    print(f"[DONE] All EDA plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
