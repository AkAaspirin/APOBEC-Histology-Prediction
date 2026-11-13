import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

OUT = Path(r"../FINAL_V2")
LABELS_DIR = OUT / "01_labels"
TABLES_DIR = OUT / "09_tables"
COMBO_DIR = OUT / "07_91_combo"
FIG_DIR = OUT / "10_figures"

# 统一字体和全局样式
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"

t1 = pd.read_csv(LABELS_DIR / "table1_labels_332.csv")
t2 = pd.read_csv(LABELS_DIR / "table2_332_label_level.csv")
t4 = pd.read_csv(TABLES_DIR / "table4_91_combo.csv")

t2 = t2[t2["label"] != "APOBEC_fraction (lr=3e-5)"]
df_a = pd.merge(t1[["label", "abs_rho_with_TMB"]], t2[["label", "rho_mean"]], on="label")

# ★ 采用 NPG (Nature Publishing Group) 风格的配色
color_map = {
    "TMB":              "#00468B", # 深蓝
    "SBS1_fraction":    "#42B540", # 草绿
    "APOBEC_fraction":  "#ED0000", # 朱红
    "APOBEC_residual":  "#925E9F", # 紫色
    "SBS5_fraction":    "#ADB6B6", # 冷灰
}

fig = plt.figure(figsize=(18, 5.5))
gs = fig.add_gridspec(1, 3, wspace=0.45)

# ================= Panel (a) =================
ax1 = fig.add_subplot(gs[0, 0])
for _, row in df_a.iterrows():
    c = color_map.get(row["label"], "black")
    ax1.scatter(row["abs_rho_with_TMB"], row["rho_mean"], color=c, s=120, zorder=3, edgecolors="none")
    offset = 0.015
    if row["label"] == "APOBEC_fraction": offset = -0.015; va="top"
    elif row["label"] == "SBS1_fraction": offset = 0.015; va="bottom"
    else: va="center"
    ax1.text(row["abs_rho_with_TMB"] + 0.02, row["rho_mean"] + offset,
             row["label"], fontsize=10, va=va, color="#333333")

ax1.set_xlabel("Correlation of label with TMB (|ρ|)", fontsize=11)
ax1.set_ylabel("Prediction performance (|ρ| with true label)", fontsize=11)
# 去掉粗体，字号稍微调小一点，防止换行
ax1.set_title("(a) Association between label predictability and TMB correlation", loc="left", fontsize=11, pad=10)
ax1.set_xlim(0, 1.05)
ax1.set_ylim(-0.05, 0.42)
ax1.grid(ls=":", lw=0.6, color="#cccccc", alpha=0.7)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ================= Panel (b) =================
ax2 = fig.add_subplot(gs[0, 1])
t4 = t4.sort_values("rho_mean", ascending=True).reset_index(drop=True)
y_pos = np.arange(len(t4))
for i, row in t4.iterrows():
    # 统一用深蓝色系，去掉花哨的颜色，线宽变细，去掉 caps (端帽) 更符合顶级期刊审美
    color = "#00468B" if "UNI" in row["combination"] else "#ED0000"
    ax2.errorbar(
        row["rho_mean"], i,
        xerr=[[row["rho_mean"] - row["ci_low"]], [row["ci_high"] - row["rho_mean"]]],
        fmt="o", color=color, capsize=0, lw=1.5, markersize=8, markeredgecolor="white", markeredgewidth=1
    )
ax2.axvline(0, color="#666666", ls="--", lw=1)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(t4["combination"], fontsize=10)
ax2.set_xlabel("Spearman ρ (95% CI, 3 seeds, n = 91)", fontsize=11)
ax2.set_title("(b) Architecture comparison (91 patients)", loc="left", fontsize=11, pad=10)
ax2.set_xlim(-0.2, 0.55)
ax2.grid(axis="x", ls=":", lw=0.6, color="#cccccc", alpha=0.7)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ================= Panel (c) =================
ax3 = fig.add_subplot(gs[0, 2])
preds, pids = [], None
for s in [42, 2024, 3407]:
    df = pd.read_csv(COMBO_DIR / f"oof_91_UNI_Trans_seed{s}.csv").drop_duplicates("patient_id").set_index("patient_id")
    if pids is None: pids = df.index.values
    else: df = df.reindex(pids)
    preds.append(df["pred"].values)
avg_pred = np.mean(preds, axis=0)

lab = pd.read_csv(r"../data/new_labels_full.csv")
lab["pid"] = lab["patient_id"].str[:12].str.upper()
lab = lab.drop_duplicates("pid").set_index("pid")
sub = lab.dropna(subset=["log1p_APOBEC_abs", "total_perMB_log"]).copy()
x = np.log(sub["total_perMB_log"] + 1); y = sub["log1p_APOBEC_abs"]
coef = np.polyfit(x, y, 1)
lab.loc[sub.index, "APOBEC_residual"] = y - np.polyval(coef, x)

resid = pd.to_numeric(lab.reindex(pids)["APOBEC_residual"], errors="coerce").values
mask = ~np.isnan(resid)

# 使用灰紫色，降低透明度，去掉描边
ax3.scatter(resid[mask], avg_pred[mask], alpha=0.5, s=20, color="#925E9F", edgecolors="none")

ax3.set_xlabel("True APOBEC residual (TMB-adjusted)", fontsize=11)
ax3.set_ylabel("Predicted APOBEC fraction", fontsize=11)
ax3.set_title("(c) Predicted score vs. TMB-adjusted residual", loc="left", fontsize=11, pad=10)
ax3.grid(ls=":", lw=0.6, color="#cccccc", alpha=0.7)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

plt.savefig(FIG_DIR / "Fig3_burden_decoupling.pdf", bbox_inches="tight", facecolor="white")
plt.savefig(FIG_DIR / "Fig3_burden_decoupling.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print(f"✅ Figure 3 (NPG Style) saved to {FIG_DIR}")