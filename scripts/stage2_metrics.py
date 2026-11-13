import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from pathlib import Path

# ============ 路径 ============
OUT = Path(r"../FINAL_V2")
MIL = Path(r"E:\Pycharm_code\mil")
OUT_DIR = OUT / "01_labels"
SEEDS = [42, 2024, 3407]
OOF_332 = MIL / "survival_data" / "oof_apobec_abs.csv"

# 332 例名单
pids_332 = set(pd.read_csv(OOF_332)["patient_id"].str[:12].str.upper())
print(f"[332 例名单] n = {len(pids_332)}")

# ============ 5 个标签的主实验 ============
RUNS = {
    "TMB":              ("tmb_run/logs", "total_perMB_log"),
    "SBS1_fraction":    ("stage5/results/B_SBS1/logs", "SBS1_fraction"),
    "APOBEC_fraction":  ("logs", "APOBEC_fraction"),
    "SBS5_fraction":    ("stage5/results/C_SBS5/logs", "SBS5_fraction"),
    "APOBEC_residual":  ("stage5/results/A_APOBEC_residual/logs", "APOBEC_residual"),
}
# 稳健性：APOBEC_fraction 的 lr=3e-5 配置
EXTRA = {
    "APOBEC_fraction (lr=3e-5)": ("original_lr3e5/logs", "APOBEC_fraction"),
}

def bootstrap_ci(y, p, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    rs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(y[idx], p[idx])
        if not np.isnan(r):
            rs.append(r)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))

def process(name, rel_dir, label_col):
    seed_rhos = []
    seed_preds = []
    y_ref = None
    pids_ref = None
    for seed in SEEDS:
        f = MIL / rel_dir / f"oof_seed{seed}.csv"
        if not f.exists():
            print(f"  [跳过] {name} seed{seed}: 文件不存在")
            continue
        df = pd.read_csv(f)
        if "pred" not in df.columns or "gt" not in df.columns:
            print(f"  [跳过] {name} seed{seed}: 列名不符 {df.columns.tolist()}")
            continue
        df["pid"] = df["patient_id"].str[:12].str.upper()
        df = df.drop_duplicates("pid").set_index("pid")
        # 过滤到 332 例
        df = df[df.index.isin(pids_332)]
        if len(df) < 300:
            print(f"  [警告] {name} seed{seed}: 过滤后仅 {len(df)} 行")
        if y_ref is None:
            y_ref = df["gt"].values
            pids_ref = df.index.values
        else:
            df = df.reindex(pids_ref)
        if df["pred"].isna().any():
            print(f"  [警告] {name} seed{seed}: reindex 后有 NaN，跳过")
            continue
        rho = spearmanr(df["gt"].values, df["pred"].values)[0]
        seed_rhos.append(rho)
        seed_preds.append(df["pred"].values)
        print(f"  seed{seed}: ρ={rho:.4f}, n={len(df)}")
    if not seed_rhos:
        return None
    # 3-seed 平均预测
    avg_pred = np.mean(seed_preds, axis=0)
    rho_avg = spearmanr(y_ref, avg_pred)[0]
    lo, hi = bootstrap_ci(y_ref, avg_pred)
    return {
        "label": name,
        "column": label_col,
        "n": len(y_ref),
        "n_seeds": len(seed_rhos),
        "rho_mean": round(float(np.mean(seed_rhos)), 4),
        "rho_sd": round(float(np.std(seed_rhos, ddof=1)), 4),
        "seed_rhos": " / ".join(f"{r:.4f}" for r in seed_rhos),
        "rho_avg_pred": round(float(rho_avg), 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "p": float(spearmanr(y_ref, avg_pred)[1]),
    }

rows = []
for name, (rel_dir, col) in RUNS.items():
    print(f"\n[{name}]  ({rel_dir})")
    r = process(name, rel_dir, col)
    if r:
        rows.append(r)

for name, (rel_dir, col) in EXTRA.items():
    print(f"\n[{name}]  ({rel_dir})")
    r = process(name, rel_dir, col)
    if r:
        rows.append(r)

out = pd.DataFrame(rows)

# ============ BH-FDR（只对非 TMB 标签做，TMB 是参照） ============
mask = out["label"] != "TMB"
if mask.sum() > 0:
    pvals = out.loc[mask, "p"].values
    reject, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    out["p_adj"] = np.nan
    out["fdr_sig"] = False
    out.loc[mask, "p_adj"] = pvals_adj
    out.loc[mask, "fdr_sig"] = reject

out_path = OUT_DIR / "table2_332_label_level.csv"
out.to_csv(out_path, index=False)

print("\n" + "=" * 110)
print("Table 2（正文）：332 例 5 标签 3-seed 汇总 + Bootstrap CI + BH-FDR")
print("=" * 110)
print(out.to_string(index=False))
print(f"\n✅ 已保存: {out_path}")