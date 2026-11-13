"""Stage 5 主训练：三个任务
  A. H&E → APOBEC_residual  (fold-internal residual)
  B. H&E → SBS1_fraction
  C. H&E → SBS5_fraction
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr, pearsonr

from dataset_stage5 import WSIBagDataset, build_splits, LABEL_CSV
from model import GatedAttentionMIL

SEEDS = [42, 2024, 3407]
N_FOLDS = 5
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25
ACCUM_STEPS = 4
MAX_PATCHES_TRAIN = 8192
MAX_PATCHES_VAL = 8192
N_SUBSAMPLE = 5
PATIENCE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT = r"../stage5\results"
HUBER_DELTA = 1.0

torch.manual_seed(42)
np.random.seed(42)


def collate_one(batch):
    return batch[0]


def evaluate(model, loader, device):
    model.eval()
    preds, gts, pids = [], [], []
    with torch.no_grad():
        for subs, y, pid in loader:
            sp = []
            for x in subs:
                x = x.to(device, non_blocking=True)
                sp.append(model(x).item())
            preds.append(np.mean(sp))
            gts.append(y.item())
            pids.append(pid)
    preds = np.array(preds); gts = np.array(gts)
    return {
        "spearman": float(spearmanr(preds, gts).correlation) if len(preds) > 2 else np.nan,
        "pearson":  float(pearsonr(preds, gts)[0]) if len(preds) > 2 else np.nan,
        "rmse":     float(np.sqrt(np.mean((preds - gts) ** 2))),
        "mae":      float(np.mean(np.abs(preds - gts))),
        "preds": preds, "gts": gts, "pids": pids,
    }


def make_stratify(df, label_col):
    q = df[label_col].quantile([1/3, 2/3]).values
    tertile = np.where(df[label_col] <= q[0], "low",
              np.where(df[label_col] >= q[1], "high", "mid"))
    combo = df["TSS"].values + "_" + tertile
    s = pd.Series(combo)
    vc = s.value_counts()
    s = s.where(s.map(vc) >= N_FOLDS, "other")
    return s.values


def train_one_fold(fold, df_tr, df_va, label_col, device, seed,
                   residual_fit=False):
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)

    if residual_fit:
        ds_tr = WSIBagDataset(df_tr, label_col=label_col,
                              max_patches=MAX_PATCHES_TRAIN, training=True,
                              residual_fit_df=df_tr)
        ds_va = WSIBagDataset(df_va, label_col=label_col,
                              max_patches=MAX_PATCHES_VAL, training=False,
                              n_subsample=N_SUBSAMPLE,
                              residual_fit_df=df_tr)
    else:
        ds_tr = WSIBagDataset(df_tr, label_col=label_col,
                              max_patches=MAX_PATCHES_TRAIN, training=True)
        ds_va = WSIBagDataset(df_va, label_col=label_col,
                              max_patches=MAX_PATCHES_VAL, training=False,
                              n_subsample=N_SUBSAMPLE)

    dl_tr = DataLoader(ds_tr, batch_size=1, shuffle=True,
                       collate_fn=collate_one, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False,
                       collate_fn=collate_one, num_workers=0)

    model = GatedAttentionMIL(in_dim=1024, hidden_dim=512, attn_dim=128,
                              n_out=1, dropout=DROPOUT).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    best_rho, best_state, best_epoch, patience_cnt = -np.inf, None, -1, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        optim.zero_grad()
        for step, (x, y, _) in enumerate(dl_tr):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).unsqueeze(0)
            loss = criterion(model(x), y) / ACCUM_STEPS
            loss.backward()
            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step(); optim.zero_grad()
            losses.append(loss.item() * ACCUM_STEPS)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step(); optim.zero_grad(); sched.step()

        val = evaluate(model, dl_va, device)
        print(f"  Epoch {epoch:03d} | loss {np.mean(losses):.4f} | "
              f"Spearman {val['spearman']:.4f}", flush=True)

        if val["spearman"] > best_rho:
            best_rho = val["spearman"]; best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (best {best_epoch})", flush=True)
                break

    model.load_state_dict(best_state)
    return model, evaluate(model, dl_va, device), best_epoch


def run_task(task_name, label_col, residual_fit=False):
    print("\n" + "=" * 70)
    print(f"任务: {task_name}  (label={label_col}, residual_fit={residual_fit})")
    print("=" * 70, flush=True)

    out_dir = os.path.join(OUT_ROOT, task_name)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    df = build_splits(label_col=label_col)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    if residual_fit:
        df = df.dropna(subset=["total_perMB_log"]).reset_index(drop=True)
    print(f"可用样本：{len(df)}", flush=True)

    strat = make_stratify(df, label_col)
    all_rows = []

    for seed in SEEDS:
        print(f"\n########## Seed {seed} ##########", flush=True)
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_p, oof_g, oof_id = [], [], []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(df, strat), start=1):
            df_tr = df.iloc[tr_idx].reset_index(drop=True)
            df_va = df.iloc[va_idx].reset_index(drop=True)
            print(f"\n=== Seed {seed} Fold {fold} "
                  f"(n_tr={len(df_tr)}, n_va={len(df_va)}) ===", flush=True)

            model, m, best_ep = train_one_fold(fold, df_tr, df_va, label_col,
                                                DEVICE, seed,
                                                residual_fit=residual_fit)
            torch.save(model.state_dict(),
                       os.path.join(out_dir, "checkpoints",
                                    f"seed{seed}_fold{fold}.pt"))
            all_rows.append({
                "seed": seed, "fold": fold, "best_epoch": best_ep,
                "spearman": m["spearman"], "pearson": m["pearson"],
                "rmse": m["rmse"], "mae": m["mae"],
            })
            oof_p.extend(m["preds"].tolist())
            oof_g.extend(m["gts"].tolist())
            oof_id.extend(m["pids"])

        oof_rho = spearmanr(oof_p, oof_g).correlation
        print(f"\n>>> Seed {seed} OOF Spearman={oof_rho:.4f}", flush=True)
        pd.DataFrame({"patient_id": oof_id, "pred": oof_p, "gt": oof_g}) \
          .to_csv(os.path.join(out_dir, "logs", f"oof_seed{seed}.csv"), index=False)

    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(out_dir, "logs", "cv_results.csv"), index=False)
    mean_rho = res["spearman"].mean()
    std_rho = res["spearman"].std()
    n_pos = int((res["spearman"] > 0).sum())
    n_tot = len(res)

    print(f"\n=== {task_name} 汇总 ===")
    print(f"Spearman: {mean_rho:.4f} ± {std_rho:.4f}")
    print(f"方向一致: {n_pos}/{n_tot}")

    decision = ("GO" if mean_rho >= 0.25 and n_pos >= 0.7 * n_tot else
                "BOUNDARY" if mean_rho >= 0.15 else "NO-GO")
    summary = {
        "task": task_name, "label_col": label_col,
        "residual_fit": residual_fit, "n_samples": len(df),
        "mean_spearman": float(mean_rho), "std_spearman": float(std_rho),
        "n_positive": n_pos, "n_total": n_tot, "decision": decision,
    }
    with open(os.path.join(out_dir, "logs", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"决策: {decision}\n")
    return summary


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    summaries = []

    # Task A: H&E → APOBEC_residual (fold-internal)
    summaries.append(run_task(
        "A_APOBEC_residual", "APOBEC_fraction", residual_fit=True))

    # Task B: H&E → SBS1_fraction
    summaries.append(run_task(
        "B_SBS1", "SBS1_fraction", residual_fit=False))

    # Task C: H&E → SBS5_fraction
    summaries.append(run_task(
        "C_SBS5", "SBS5_fraction", residual_fit=False))

    # 汇总
    print("\n" + "=" * 70)
    print("Stage 5 最终汇总")
    print("=" * 70)
    summary_df = pd.DataFrame(summaries)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUT_ROOT, "stage5_summary.csv"), index=False)
    print(f"\n已保存: {os.path.join(OUT_ROOT, 'stage5_summary.csv')}")


if __name__ == "__main__":
    main()