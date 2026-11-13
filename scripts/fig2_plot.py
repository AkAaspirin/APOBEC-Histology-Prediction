import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

# ============ 路径 ============
OUT = Path(r"../FINAL_V2")
LABELS_DIR = OUT / "01_labels"
FIG_DIR = OUT / "10_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MIL = Path(r"E:\Pycharm_code\mil")
SEEDS = [42, 2024, 3407]

# ============ 全局样式 (NPG 风格) ============
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"

# ============ 读取数据 ============
t2 = pd.read_csv(LABELS_DIR / "table2_332_label_level.csv")
t2 = t2[t2["label"] != "APOBEC_fraction (lr=3e-5)"]
t2 = t2.sort_values("rho_mean", ascending=False).reset_index(drop=True)

# NPG 配色
color_map = {
    "TMB": "#00468B",  # 深蓝
    "SBS1_fraction": "#42B540",  # 草绿
    "APOBEC_fraction": "#ED0000",  # 朱红
    "APOBEC_residual": "#925E9F",  # 紫色
    "SBS5_fraction": "#ADB6B6",  # 冷灰
}


def load_avg_pred(rel_dir):
    preds, gt, pids = [], None, None
    for s in SEEDS:
        f = MIL / rel_dir / f"oof_seed{s}.csv"
        if not f.exists():
            print(f"[跳过] {f}")
            continue
        df = pd.read_csv(f).drop_duplicates("patient_id").set_index("patient_id")
        if gt is None:
            gt = df["gt"].values
            pids = df.index.values
        else:
            df = df.reindex(pids)
        preds.append(df["pred"].values)
    return np.mean(preds, axis=0), gt


# ============ 画图 ============
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.38, wspace=0.28)

# ----- Panel (a): 水平森林图 -----
ax1 = fig.add_subplot(gs[0, :])
y_pos = np.arange(len(t2))
for i, row in t2.iterrows():
    c = color_map.get(row["label"], "black")
    ax1.errorbar(
        row["rho_mean"], i,
        xerr=[[row["rho_mean"] - row["ci_low"]], [row["ci_high"] - row["rho_mean"]]],
        fmt="o", color=c, capsize=3, lw=1.5, markersize=8, markeredgecolor="white", markeredgewidth=1
    )
    if str(row["fdr_sig"]).lower() == "true":
        ax1.text(row["ci_high"] + 0.015, i, "*", va="center", fontsize=16, fontweight="bold")
ax1.axvline(0, color="#666666", ls="--", lw=1)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(t2["label"], fontsize=11)
ax1.set_xlabel("Spearman ρ (mean across 3 seeds, with patient-level bootstrap 95% CI, n = 332)", fontsize=11)
ax1.set_xlim(-0.2, 0.55)
ax1.set_title("(a) Label-level predictability", loc="left", fontsize=11, pad=10)
ax1.grid(axis="x", ls=":", lw=0.6, color="#cccccc", alpha=0.7)
ax1.invert_yaxis()
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ----- Panel (b) & (c): 散点图 -----
panels = [
    ("tmb_run/logs", "(b) TMB (burden itself) - Ensemble Prediction", "#00468B",
     "True TMB (log)", "Predicted TMB (3-seed avg)", gs[1, 0]),
    ("stage5/results/C_SBS5/logs", "(c) SBS5 (process) - Ensemble Prediction", "#ADB6B6",
     "True SBS5 fraction", "Predicted SBS5 (3-seed avg)", gs[1, 1]),
]

for rel_dir, title, color, xlabel, ylabel, pos in panels:
    pred, gt = load_avg_pred(rel_dir)
    ax = fig.add_subplot(pos)

    # 画散点
    ax.scatter(gt, pred, alpha=0.4, s=15, color=color, edgecolors="none")

    # 计算并画线性拟合红线
    z = np.polyfit(gt, pred, 1)
    xs = np.linspace(gt.min(), gt.max(), 100)
    ax.plot(xs, np.polyval(z, xs), color="#ED0000", lw=1.5, alpha=0.9)

    # 计算集成相关系数
    rho, _ = spearmanr(gt, pred)
    ax.set_title(f"{title}\nEnsemble ρ = {rho:.3f}", loc="left", fontsize=10, pad=10)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(ls=":", lw=0.6, color="#cccccc", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ============ 保存 ============
plt.savefig(FIG_DIR / "Fig2_label_level.pdf", bbox_inches="tight", facecolor="white")
plt.savefig(FIG_DIR / "Fig2_label_level.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print(f"✅ Figure 2 (with Ensemble labels and red fit lines) saved to {FIG_DIR}")