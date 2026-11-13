"""Stage 5 dataset：支持 fold-internal residual 拟合。"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.linear_model import LinearRegression

PT_DIR = Path(r"E:\PAPER\project_apobec_pathomics\02_wsi\blca\uni\pt_files")
PT_BARCODE_CSV = Path(r"E:\PAPER\project_apobec_pathomics\03_mil\pt_files_barcodes.csv")
LABEL_CSV = Path(r"../stage5\stage5_labels.csv")

SEED = 42


def build_splits(label_csv=LABEL_CSV, pt_barcode_csv=PT_BARCODE_CSV,
                 label_col="APOBEC_fraction"):
    df_label = pd.read_csv(label_csv)
    df_label["patient_id"] = df_label["Tumor_Sample_Barcode"].str[:12]
    df_label["TSS"] = df_label["Tumor_Sample_Barcode"].str[5:7]

    df_pt = pd.read_csv(pt_barcode_csv)
    n_wsi = df_pt.groupby("patient_id").size().rename("n_wsi")

    common = set(df_label["patient_id"]) & set(df_pt["patient_id"])
    df = df_label[df_label["patient_id"].isin(common)].copy()
    df = df.merge(n_wsi, on="patient_id", how="left")

    q = df[label_col].quantile([1 / 3, 2 / 3]).values
    df["tertile"] = np.where(df[label_col] <= q[0], "low",
                     np.where(df[label_col] >= q[1], "high", "mid"))

    keep = ["patient_id", "Tumor_Sample_Barcode", label_col, "TSS",
            "n_wsi", "tertile", "total_perMB_log", "total_perMB"]
    return df[keep].reset_index(drop=True)


class WSIBagDataset(Dataset):
    """支持 fold-internal residual 的 WSI dataset。"""

    def __init__(self, df, pt_dir=PT_DIR, pt_barcode_csv=PT_BARCODE_CSV,
                 label_col="APOBEC_fraction",
                 max_patches=8192, training=True, n_subsample=5,
                 residual_fit_df=None, covar_col="total_perMB_log",
                 residual_source_col="APOBEC_fraction"):
        """
        residual_fit_df: 若非 None，用这个 df 拟合 residual 模型，
                        再对 df 计算残差作为新 label。
        """
        self.df = df.reset_index(drop=True).copy()
        self.pt_dir = Path(pt_dir)
        self.max_patches = max_patches
        self.training = training
        self.n_subsample = n_subsample

        # fold-internal residual
        if residual_fit_df is not None:
            lr = LinearRegression().fit(
                residual_fit_df[[covar_col]].values,
                residual_fit_df[residual_source_col].values,
            )
            pred = lr.predict(self.df[[covar_col]].values)
            self.df["_residual"] = self.df[residual_source_col].values - pred
            self.label_col = "_residual"
        else:
            self.label_col = label_col

        df_pt = pd.read_csv(pt_barcode_csv)
        df_pt["slide_id"] = df_pt["pt_file"].str.replace(".pt", "", regex=False)
        self.pt_map = df_pt.groupby("patient_id").apply(
            lambda g: list(zip(g["pt_file"], g["slide_id"]))
        ).to_dict()

    def __len__(self):
        return len(self.df)

    def _load_bag(self, pid):
        items = self.pt_map.get(pid, [])
        feats, slide_ids = [], []
        for pt_file, slide_id in items:
            f = torch.load(self.pt_dir / pt_file, map_location="cpu")
            feats.append(f)
            slide_ids.extend([slide_id] * f.shape[0])
        x = torch.cat(feats, dim=0)
        return x, slide_ids

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["patient_id"]
        x, slide_ids = self._load_bag(pid)
        y = torch.tensor(row[self.label_col], dtype=torch.float32)

        N = x.shape[0]
        if self.training:
            if self.max_patches and N > self.max_patches:
                perm = torch.randperm(N)[:self.max_patches]
                x = x[perm]
            return x, y, pid

        if self.max_patches and N > self.max_patches:
            subs = []
            for k in range(self.n_subsample):
                g = torch.Generator().manual_seed(SEED + k)
                perm = torch.randperm(N, generator=g)[:self.max_patches]
                subs.append(x[perm])
            return subs, y, pid
        else:
            return [x], y, pid


if __name__ == "__main__":
    df = build_splits()
    print(f"可用样本：{len(df)}")
    print(df.head())
    ds = WSIBagDataset(df, training=False)
    x_list, y, pid = ds[0]
    print(f"\n样本 {pid}: {len(x_list)} subsample, shape={x_list[0].shape}, label={y.item():.4f}")