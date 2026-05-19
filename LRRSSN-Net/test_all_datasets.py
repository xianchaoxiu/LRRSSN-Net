import argparse
import csv
import glob
import os
import time

import h5py
import numpy as np
import scipy.io as sio
import torch

from model_opt.main_net3 import DeepUnfoldingSSN
from tool import evaluate_clustering


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_mat_contents(file_path):
    try:
        return sio.loadmat(file_path)
    except NotImplementedError as exc:
        if "Please use HDF reader for matlab v7.3 files" not in str(exc):
            raise
        with h5py.File(file_path, "r") as data:
            return {key: np.array(data[key]) for key in data.keys()}


def _pick_first_available(data, candidates, kind):
    for key in candidates:
        if key in data:
            return data[key], key
    raise KeyError(f"Missing {kind} key. Tried: {candidates}")


def load_data(file_path, device):
    data = _load_mat_contents(file_path)
    X_raw, x_key = _pick_first_available(data, ["fea", "X", "test_X"], "feature")
    y_raw, y_key = _pick_first_available(data, ["gnd", "Y", "test_labels"], "label")
    X = X_raw
    labels = np.asarray(y_raw).reshape(-1)

    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    if X.shape[0] == labels.shape[0]:
        X = X.T

    if X.shape[1] != labels.shape[0]:
        raise ValueError(
            f"Sample count mismatch after transpose check: X.shape={X.shape}, labels={labels.shape}, x_key={x_key}, y_key={y_key}"
        )

    valid = np.isfinite(labels)
    if not np.all(valid):
        X = X[:, valid]
        labels = labels[valid]

    if labels.size == 0:
        raise ValueError("All labels are invalid after filtering")

    labels = labels.astype(np.int64, copy=False)

    X_tensor = torch.from_numpy(X).to(device)
    X_tensor = X_tensor - torch.mean(X_tensor, dim=1, keepdim=True)
    X_tensor = X_tensor / (torch.norm(X_tensor, p=2, dim=0, keepdim=True) + 1e-8)

    n_clusters = int(len(np.unique(labels)))
    if n_clusters <= 1:
        raise ValueError(f"Invalid number of clusters: {n_clusters}")

    return X_tensor, labels, n_clusters


def resolve_checkpoint(dataset_name, checkpoint_dir, checkpoint_path):
    if checkpoint_path:
        path = checkpoint_path if os.path.isabs(checkpoint_path) else os.path.join(SCRIPT_DIR, checkpoint_path)
        return os.path.abspath(path)

    ckpt_name = f"ssn_{dataset_name}_best.pth"
    path = os.path.join(checkpoint_dir, ckpt_name)
    return os.path.abspath(path)


def run_single_dataset(dataset_path, checkpoint_path, device, n_stages, kernel_size, warmup):
    X, labels, n_clusters = load_data(dataset_path, device)
    d, n = X.shape

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    if n_stages is None:
        n_stages = ckpt.get("n_stages", 1)
    if kernel_size is None:
        kernel_size = ckpt.get("kernel_size", 7)

    model = DeepUnfoldingSSN(
        d,
        n,
        num_stages=n_stages,
        kernel_size=kernel_size,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    amp_enabled = device.type == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                _ = model(X)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            Z, _, _, J = model(X)[:4]

        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_time_s = time.perf_counter() - t0

    res_z = evaluate_clustering(Z, labels, n_clusters, show_plot=False)
    res_j = evaluate_clustering(J, labels, n_clusters, show_plot=False)

    return {
        "status": "ok",
        "num_features": int(d),
        "num_samples": int(n),
        "n_clusters": int(n_clusters),
        "n_stages": int(n_stages),
        "kernel_size": int(kernel_size),
        "infer_time_s": float(infer_time_s),
        "z_acc": float(res_z["ACC"]),
        "z_nmi": float(res_z["NMI"]),
        "z_ari": float(res_z["ARI"]),
        "j_acc": float(res_j["ACC"]),
        "j_nmi": float(res_j["NMI"]),
        "j_ari": float(res_j["ARI"]),
        "error": "",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Batch test all datasets and save metrics to CSV.")
    parser.add_argument("--datasets-dir", default="datasets", help="Directory that contains .mat datasets")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory of per-dataset checkpoints")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ssn_USPS_PART_best.pth",
        help="Single checkpoint used for all datasets",
    )
    parser.add_argument("--n-stages", type=int, default=None, help="Override stage count; default reads from checkpoint")
    parser.add_argument("--kernel-size", type=int, default=None, help="Override kernel size; default reads from checkpoint")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup forwards before timing")
    parser.add_argument("--csv-path", default="all_datasets_test_results.csv", help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets_dir = args.datasets_dir
    if not os.path.isabs(datasets_dir):
        datasets_dir = os.path.join(SCRIPT_DIR, datasets_dir)
    checkpoint_dir = args.checkpoint_dir
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(SCRIPT_DIR, checkpoint_dir)
    csv_path = args.csv_path
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(SCRIPT_DIR, csv_path)

    dataset_paths = sorted(glob.glob(os.path.join(datasets_dir, "*.mat")))
    if not dataset_paths:
        raise FileNotFoundError(f"No .mat files found in {datasets_dir}")

    fieldnames = [
        "dataset",
        "checkpoint",
        "status",
        "num_features",
        "num_samples",
        "n_clusters",
        "n_stages",
        "kernel_size",
        "infer_time_s",
        "z_acc",
        "z_nmi",
        "z_ari",
        "j_acc",
        "j_nmi",
        "j_ari",
        "error",
    ]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    print(f"Device: {device}")
    print(f"Datasets: {len(dataset_paths)} found in {datasets_dir}")
    print(f"Checkpoint for all datasets: {os.path.abspath(resolve_checkpoint('unused', checkpoint_dir, args.checkpoint))}")
    print(f"CSV: {csv_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_path in dataset_paths:
            dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
            checkpoint_path = resolve_checkpoint(dataset_name, checkpoint_dir, args.checkpoint)

            base_row = {
                "dataset": dataset_name,
                "checkpoint": checkpoint_path,
                "status": "",
                "num_features": "",
                "num_samples": "",
                "n_clusters": "",
                "n_stages": "",
                "kernel_size": "",
                "infer_time_s": "",
                "z_acc": "",
                "z_nmi": "",
                "z_ari": "",
                "j_acc": "",
                "j_nmi": "",
                "j_ari": "",
                "error": "",
            }

            if not os.path.exists(checkpoint_path):
                row = dict(base_row)
                row["status"] = "missing_checkpoint"
                row["error"] = "checkpoint_not_found"
                writer.writerow(row)
                print(f"[SKIP] {dataset_name}: checkpoint not found -> {checkpoint_path}")
                continue

            try:
                result = run_single_dataset(
                    dataset_path=dataset_path,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    n_stages=args.n_stages,
                    kernel_size=args.kernel_size,
                    warmup=max(0, args.warmup),
                )
                row = dict(base_row)
                row.update(result)
                writer.writerow(row)
                print(
                    f"[OK] {dataset_name} | t={row['infer_time_s']:.4f}s | "
                    f"Z-ACC={row['z_acc']:.4f} Z-NMI={row['z_nmi']:.4f}"
                )
            except Exception as exc:
                row = dict(base_row)
                row["status"] = "failed"
                row["error"] = str(exc).replace("\n", " | ")
                writer.writerow(row)
                print(f"[FAIL] {dataset_name}: {exc}")

            f.flush()

    print("Done.")


if __name__ == "__main__":
    main()
