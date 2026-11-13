import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

OUT = Path(r"../FINAL_V2")
TABLES_DIR = OUT / "09_tables"
FIG_DIR = OUT / "10_figures"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.linewidth"] = 0.8

# 采用 Nature 风格配色
COLOR_HIGH = "#ED0000" # 朱红
COLOR_LOW = "#00468B"  # 深蓝

fig = plt.figure(figsize=(14, 5.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.35)

# ================= Panel (a): 参考 APOBEC KM =================
ax1 = fig.add_subplot(gs[0, 0])

lab = pd.read_csv(r"../data/new_labels_full.csv")
oof = pd.read_csv(r"../data/oof_apobec_abs.csv")
pids_332 = set(oof["patient_id"].str[:12].str.upper())

lab["pid"] = lab["patient_id"].str[:12].str.upper()
lab = lab[lab["pid"].isin(pids_332)]
lab = lab.dropna(subset=["OS_time", "OS_event", "log1p_APOBEC_abs"])
lab = lab[lab["OS_time"] > 0]

med = lab["log1p_APOBEC_abs"].median()
lab["grp"] = np.where(lab["log1p_APOBEC_abs"] > med, "High", "Low")

kmf_high = KaplanMeierFitter()
kmf_low = KaplanMeierFitter()

mask_high = lab["grp"] == "High"
mask_low = lab["grp"] == "Low"

kmf_high.fit(lab.loc[mask_high, "OS_time"], lab.loc[mask_high, "OS_event"], label="High APOBEC")
kmf_low.fit(lab.loc[mask_low, "OS_time"], lab.loc[mask_low, "OS_event"], label="Low APOBEC")

kmf_high.plot_survival_function(ax=ax1, color=COLOR_HIGH, ci_show=False, lw=1.8)
kmf_low.plot_survival_function(ax=ax1, color=COLOR_LOW, ci_show=False, lw=1.8)

res = logrank_test(lab.loc[mask_high, "OS_time"], lab.loc[mask_low, "OS_time"],
                   event_observed_A=lab.loc[mask_high, "OS_event"],
                   event_observed_B=lab.loc[mask_low, "OS_event"])
p_val = res.p_value

t6 = pd.read_csv(TABLES_DIR / "table6_survival_ref.csv")
hr = t6["HR"].values[0]
ci_low = t6["CI_low"].values[0]
ci_high = t6["CI_high"].values[0]

# 标题分两行写，更加清爽
ax1.set_title(f"(a) Reference absolute APOBEC burden\nHR = {hr:.3f} (95% CI {ci_low:.3f}–{ci_high:.3f}), log-rank p = {p_val:.4f}",
              loc="left", fontsize=11, pad=10)
ax1.set_xlabel("Time (days)", fontsize=11)
ax1.set_ylabel("Overall survival probability", fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.grid(ls=":", lw=0.6, color="#cccccc", alpha=0.7)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(loc="lower left", frameon=False, fontsize=10)

# ================= Panel (b): 91 例 4 组合森林图 =================
ax2 = fig.add_subplot(gs[0, 1])
t5 = pd.read_csv(TABLES_DIR / "table6_survival_91_combo.csv")
t5 = t5.sort_values("HR", ascending=True).reset_index(drop=True)

y_pos = np.arange(len(t5))
for i, row in t5.iterrows():
    color = COLOR_LOW if "UNI" in row["combination"] else COLOR_HIGH
    # ★ 关键修复：去掉 caps 使图形更干净
    ax2.errorbar(
        row["HR"], i,
        xerr=[[row["HR"] - row["CI_low"]], [row["CI_high"] - row["HR"]]],
        fmt="o", color=color, capsize=0, lw=1.5, markersize=8,
        markeredgecolor="white", markeredgewidth=1
    )

ax2.axvline(1, color="#666666", ls="--", lw=1)
# ★ 核心修复：恢复对数坐标，并扩大范围以容纳 0.059 - 18.058
ax2.set_xscale("log")
ax2.set_xlim(0.03, 30)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(t5["combination"], fontsize=10)
ax2.set_xlabel("Hazard Ratio (log scale, 95% CI, n = 91)", fontsize=11)
ax2.set_title("(b) Predicted scores (91 patients)", loc="left", fontsize=11, pad=10)
ax2.grid(axis="x", ls=":", lw=0.6, color="#cccccc", alpha=0.7)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.savefig(FIG_DIR / "Fig4_survival.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(FIG_DIR / "Fig4_survival.pdf", bbox_inches="tight", facecolor="white")
plt.close()

print(f"✅ Figure 4 (NPG Style) saved to {FIG_DIR}")