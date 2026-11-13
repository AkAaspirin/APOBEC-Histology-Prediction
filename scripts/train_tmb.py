import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr, pearsonr

from dataset_original import WSIBagDataset, build_splits
from model import GatedAttentionMIL


# ---------------- 配置 ----------------
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

LABEL_COL = "total_perMB_log"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = r"../tmb_run"
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
            sub_preds = []
            for x in subs:
                x = x.to(device, non_blocking=True)
                sub_preds.append(model(x).item())
            preds.append(np.mean(sub_preds))
            gts.append(y.item())
            pids.append(pid)
    preds = np.array(preds)
    gts = np.array(gts)
    return {
        "spearman": float(spearmanr(preds, gts).correlation) if len(preds) > 2 else np.nan,
        "pearson":  float(pearsonr(preds, gts)[0]) if len(preds) > 2 else np.nan,
        "rmse":     float(np.sqrt(np.mean((preds - gts) ** 2))),
        "mae":      float(np.mean(np.abs(preds - gts))),
        "preds": preds, "gts": gts, "pids": pids,
    }


def make_stratify_labels(df):
    combo = df["TSS"] + "_" + df["tertile"]
    vc = combo.value_counts()
    combo = combo.where(combo.map(vc) >= N_FOLDS, "other")
    return combo.values


def train_one_fold(fold, df_tr, df_va, device, seed):
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)

    ds_tr = WSIBagDataset(df_tr, label_col=LABEL_COL,
                          max_patches=MAX_PATCHES_TRAIN, training=True)
    ds_va = WSIBagDataset(df_va, label_col=LABEL_COL,
                          max_patches=MAX_PATCHES_VAL, training=False,
                          n_subsample=N_SUBSAMPLE)
    dl_tr = DataLoader(ds_tr, batch_size=1, shuffle=True,
                       collate_fn=collate_one, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False,
                       collate_fn=collate_one, num_workers=0)

    model = GatedAttentionMIL(in_dim=1024, hidden_dim=512,
                              attn_dim=128, n_out=1,
                              dropout=DROPOUT).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR,
                              weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    best_rho = -np.inf
    best_state = None
    best_epoch = -1
    patience_cnt = 0

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
                optim.step()
                optim.zero_grad()
            losses.append(loss.item() * ACCUM_STEPS)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        optim.zero_grad()
        sched.step()

        val = evaluate(model, dl_va, device)
        print(f"  Epoch {epoch:03d} | loss {np.mean(losses):.4f} | "
              f"RMSE {val['rmse']:.4f} | Spearman {val['spearman']:.4f}",
              flush=True)

        if val["spearman"] > best_rho:
            best_rho = val["spearman"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (best {best_epoch})",
                      flush=True)
                break

    model.load_state_dict(best_state)
    final = evaluate(model, dl_va, device)
    return model, final, best_epoch


def main():
    os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "logs"), exist_ok=True)

    df = build_splits(label_col=LABEL_COL)
    print(f"可用样本：{len(df)}")
    print(f"标签分布：mean={df[LABEL_COL].mean():.4f}, "
          f"std={df[LABEL_COL].std():.4f}, "
          f"min={df[LABEL_COL].min():.4f}, "
          f"max={df[LABEL_COL].max():.4f}\n")

    strat = make_stratify_labels(df)

    all_fold_rows = []
    for seed in SEEDS:
        print(f"\n########## Seed {seed} ##########", flush=True)
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_preds, oof_gts, oof_pids = [], [], []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(df, strat), start=1):
            df_tr = df.iloc[tr_idx].reset_index(drop=True)
            df_va = df.iloc[va_idx].reset_index(drop=True)
            print(f"\n=== Seed {seed} Fold {fold} "
                  f"(n_tr={len(df_tr)}, n_va={len(df_va)}) ===", flush=True)

            model, metrics, best_epoch = train_one_fold(fold, df_tr, df_va,
                                                         DEVICE, seed)
            torch.save(model.state_dict(),
                       os.path.join(OUT_DIR, "checkpoints",
                                    f"seed{seed}_fold{fold}_best.pt"))

            all_fold_rows.append({
                "seed": seed, "fold": fold, "best_epoch": best_epoch,
                "spearman": metrics["spearman"], "pearson": metrics["pearson"],
                "rmse": metrics["rmse"], "mae": metrics["mae"],
                "n_val": len(df_va),
            })
            oof_preds.extend(metrics["preds"].tolist())
            oof_gts.extend(metrics["gts"].tolist())
            oof_pids.extend(metrics["pids"])

        oof_rho = spearmanr(oof_preds, oof_gts).correlation
        oof_r = pearsonr(oof_preds, oof_gts)[0]
        print(f"\n>>> Seed {seed} OOF: Spearman={oof_rho:.4f}, "
              f"Pearson={oof_r:.4f}", flush=True)
        pd.DataFrame({"patient_id": oof_pids, "pred": oof_preds, "gt": oof_gts}) \
          .to_csv(os.path.join(OUT_DIR, "logs", f"oof_seed{seed}.csv"),
                  index=False)

    res = pd.DataFrame(all_fold_rows)
    res.to_csv(os.path.join(OUT_DIR, "logs", "cv_results.csv"), index=False)
    print("\n=== TMB (log) 汇总 ===")
    print(res.to_string(index=False))
    print(f"\nSpearman: {res['spearman'].mean():.4f} ± {res['spearman'].std():.4f}")
    print(f"RMSE:     {res['rmse'].mean():.4f} ± {res['rmse'].std():.4f}")
    print(f"MAE:      {res['mae'].mean():.4f} ± {res['mae'].std():.4f}")

    mean_rho = res["spearman"].mean()
    n_pos = (res["spearman"] > 0).sum()
    n_total = len(res)
    if mean_rho >= 0.25 and n_pos >= 0.7 * n_total:
        decision = "GO"
    elif mean_rho >= 0.15:
        decision = "BOUNDARY"
    else:
        decision = "NO-GO"

    summary = {
        "n_samples": int(len(df)),
        "label_col": LABEL_COL,
        "mean_spearman": float(mean_rho),
        "std_spearman": float(res["spearman"].std()),
        "mean_rmse": float(res["rmse"].mean()),
        "mean_mae": float(res["mae"].mean()),
        "n_fold_positive": int(n_pos),
        "n_fold_total": int(n_total),
        "decision": decision,
    }
    with open(os.path.join(OUT_DIR, "logs", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n决策：{decision}")
    print(f"已保存到 {OUT_DIR}/logs/")


if __name__ == "__main__":
    main()