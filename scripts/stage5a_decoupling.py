import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

OUT = Path(r"../FINAL_V2")
IN_DIR = OUT / "07_91_combo"
OUT_DIR = OUT / "09_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAB_CSV = r"../data/new_labels_full.csv"
OOF_332 = r"../data/oof_apobec_abs.csv"
SEEDS = [42, 2024, 3407]
COMBOS = [("UNI", "Gate"), ("UNI", "Trans"), ("Phikon", "Gate"), ("Phikon", "Trans")]

# ============ 读标签，算 APOBEC_residual（332 例内） ============
pids_332 = set(pd.read_csv(OOF_332)["patient_id"].str[:12].str.upper())
lab = pd.read_csv(LAB_CSV)
lab["pid"] = lab["patient_id"].str[:12].str.upper()
lab = lab.drop_duplicates("pid").set_index("pid")
lab = lab[lab.index.isin(pids_332)]
sub = lab.dropna(subset=["log1p_APOBEC_abs", "total_perMB_log"]).copy()
x = np.log(sub["total_perMB_log"] + 1)
y = sub["log1p_APOBEC_abs"]
coef = np.polyfit(x, y, 1)
lab.loc[sub.index, "APOBEC_residual"] = y - np.polyval(coef, x)

# ============ 91 例：4 组合的 3-seed 平均预测 vs TMB / residual / fraction ============
rows = []
for enc, agg in COMBOS:
    seed_preds = []
    y_frac = None
    pids_ref = None
    for seed in SEEDS:
        f = IN_DIR / f"oof_91_{enc}_{agg}_seed{seed}.csv"
        if not f.exists():
            print(f"[缺失] {f}")
            continue
        df = pd.read_csv(f).drop_duplicates("patient_id").set_index("patient_id")
        if y_frac is None:
            y_frac = df["gt"].values
            pids_ref = df.index.values
        else:
            df = df.reindex(pids_ref)
        if df["pred"].isna().any():
            continue
        seed_preds.append(df["pred"].values)
    if not seed_preds:
        continue
    avg_pred = np.mean(seed_preds, axis=0)
    # 取这些患者的 TMB 和 residual
    sub91 = lab.reindex(pids_ref)
    tmb = pd.to_numeric(sub91["total_perMB_log"], errors="coerce").values
    resid = pd.to_numeric(sub91["APOBEC_residual"], errors="coerce").values
    frac = pd.to_numeric(sub91["APOBEC_fraction"], errors="coerce").values

    def safe_rho(a, b):
        m = ~np.isnan(a) & ~np.isnan(b)
        if m.sum() < 10: return np.nan, np.nan, int(m.sum())
        r, p = spearmanr(a[m], b[m])
        return float(r), float(p), int(m.sum())

    r_tmb, p_tmb, n_tmb = safe_rho(avg_pred, tmb)
    r_res, p_res, n_res = safe_rho(avg_pred, resid)
    r_frc, p_frc, n_frc = safe_rho(avg_pred, frac)

    rows.append({
        "combination": f"{enc} + {agg}MIL",
        "n": len(avg_pred),
        "rho_pred_vs_TMB": round(r_tmb, 4) if not np.isnan(r_tmb) else np.nan,
        "p_TMB": p_tmb,
        "rho_pred_vs_residual": round(r_res, 4) if not np.isnan(r_res) else np.nan,
        "p_residual": p_res,
        "rho_pred_vs_fraction": round(r_frc, 4) if not np.isnan(r_frc) else np.nan,
        "p_fraction": p_frc,
    })

out = pd.DataFrame(rows)
out_path = OUT_DIR / "table5_burden_decoupling_91.csv"
out.to_csv(out_path, index=False)

print("=" * 100)
print("Table S4（补充）：91 例 4 组合的负荷解耦分析")
print("=" * 100)
print(out.to_string(index=False))
print(f"\n✅ 已保存: {out_path}")