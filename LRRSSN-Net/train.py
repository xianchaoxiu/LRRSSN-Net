import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import h5py
from model_opt.main_net3 import DeepUnfoldingSSN
from tool import (
    evaluate_clustering,
    plot_eigengap,
    plot_tsne,
    set_seed,
    visualize_stage_evolution,
)

set_seed(seed=42)
def block_diagonal_loss(Z, k):
    W = (torch.abs(Z) + torch.abs(Z.t())) / 2
    D = torch.diag(torch.sum(W, dim=1))
    L = D - W
    eigvals = torch.linalg.eigvalsh(L + 1e-6 * torch.eye(L.shape[0], device=L.device))
    return torch.sum(eigvals[:k])


def _load_mat_contents(file_path):
    try:
        return sio.loadmat(file_path)
    except NotImplementedError as exc:
        if "Please use HDF reader for matlab v7.3 files" not in str(exc):
            raise
        with h5py.File(file_path, "r") as data:
            return {key: np.array(data[key]) for key in data.keys()}

def load_data(file_path, device, pca_components=None):
    data = _load_mat_contents(file_path)
    X = data["fea"] if "fea" in data else data["X"]
    labels = np.asarray(data["gnd"] if "gnd" in data else data["Y"]).reshape(-1)
    if X.shape[0] == labels.shape[0]:
        X = X.T
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    if pca_components is not None:
        x_samples = X.T  # (n_samples, n_features)
        original_dim = x_samples.shape[1]
        max_valid_components = min(x_samples.shape[0], x_samples.shape[1])
        if isinstance(pca_components, int):
            n_components = min(pca_components, max_valid_components)
        else:
            n_components = pca_components

        pca = PCA(n_components=n_components, svd_solver="full", random_state=42)
        x_samples = pca.fit_transform(x_samples)
        reduced_dim = x_samples.shape[1]
        explained = float(np.sum(pca.explained_variance_ratio_))
        print(
            f"PCA applied: {original_dim} -> {reduced_dim} dims, explained variance={explained:.4f}"
        )
        X = x_samples.T.astype(np.float32, copy=False)

    labels = labels.astype(np.int64, copy=False)
    X_tensor = torch.from_numpy(X).to(device)
    X_tensor = X_tensor - torch.mean(X_tensor, dim=1, keepdim=True)
    X_tensor = X_tensor / (torch.norm(X_tensor, p=2, dim=0, keepdim=True) + 1e-8)
    return X_tensor, labels, len(np.unique(labels))


def run_experiment(
    data_path="./datasets/MSTAR_SOC_CNN.mat",
    n_stages=1,
    epochs=600,
    save_model=True,
    ckpt_dir="./checkpoints",
    pca_components=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, labels, n_clusters = load_data(data_path, device, pca_components=pca_components)
    d, n = X.shape

    model = DeepUnfoldingSSN(
        d,
        n,
        num_stages=n_stages,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0

    print(
        f"Start training {data_path} | Device: {device} | "
        f"dictionary=X"
    )

    total_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(X)
            Z, E, all_FZ, J = outputs[:4]
            loss_recon = torch.norm(X - torch.matmul(X, Z) - E, p="fro") ** 2
            loss_BDR = block_diagonal_loss(Z, n_clusters)
            loss_BDR_J = torch.norm(Z - J, p="fro") ** 2

            loss_fun = 0
            for fz in all_FZ:
                loss_fun += torch.norm(fz, p=1)
            loss_resid = loss_fun / len(all_FZ)

            loss_sparse_E = torch.norm(E, p=1)
            total_loss = (
                loss_recon
                + 1.0 * loss_BDR
                + 0.01 * loss_sparse_E
                + 0.01 * loss_resid
            )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        current_epoch = epoch + 1
        if current_epoch % 20 == 0 or current_epoch == 1:
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    val_outputs = model(X)
                    val_Z, _, _, val_J = val_outputs[:4]
                res = evaluate_clustering(val_Z, labels, n_clusters, show_plot=False)
                res_j = evaluate_clustering(val_J, labels, n_clusters, show_plot=False)

                print(
                    f"Epoch {current_epoch:3d}/{epochs} | Loss: {total_loss.item():.4f} | "
                    f"Z-ACC: {res['ACC']:.4f} | Z-NMI: {res['NMI']:.4f} | Z-ARI: {res['ARI']:.4f} | "
                    f"J-ACC: {res_j['ACC']:.4f} | J-NMI: {res_j['NMI']:.4f} | J-ARI: {res_j['ARI']:.4f}"
                )

                if res["ACC"] > best_acc:
                    best_acc = res["ACC"]
                    best_epoch = current_epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

    total_end_time = time.time()

    print(f"\nTraining done. Best epoch: {best_epoch}, Best ACC: {best_acc:.4f}")

    checkpoint_path = None
    if save_model:
        os.makedirs(ckpt_dir, exist_ok=True)
        dataset_name = os.path.splitext(os.path.basename(data_path))[0]
        checkpoint_path = os.path.join(ckpt_dir, f"ssn_{dataset_name}_best.pth")
        torch.save(
            {
                "model_state_dict": best_model_wts,
                "data_path": data_path,
                "n_stages": n_stages,
                "best_epoch": best_epoch,
                "best_acc": best_acc,
            },
            checkpoint_path,
        )
        print(f"Saved best model to: {checkpoint_path}")

    model.load_state_dict(best_model_wts)
    model.eval()

    with torch.no_grad():
        infer_start = time.time()
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            final_outputs = model(X)
            final_Z, final_E, _, final_J = final_outputs[:4]
        infer_end = time.time()

        final_res = evaluate_clustering(final_Z, labels, n_clusters, show_plot=True)
        final_res_j = evaluate_clustering(final_J, labels, n_clusters, show_plot=False)

        print("-" * 30)
        print("Final evaluation (best model)")
        print(f"Z-ACC: {final_res['ACC']:.4f}")
        print(f"Z-NMI: {final_res['NMI']:.4f}")
        print(f"J-ACC: {final_res_j['ACC']:.4f}")
        print(f"J-NMI: {final_res_j['NMI']:.4f}")
        print(f"Single inference time: {infer_end - infer_start:.4f} s")
        print(f"Total training time: {total_end_time - total_start_time:.2f} s")
        print("-" * 30)

    return {
        "best_epoch": best_epoch,
        "best_acc": best_acc,
        "checkpoint_path": checkpoint_path,
        "final_res_z": final_res,
        "final_res_j": final_res_j,
    }
if __name__ == "__main__":
    run_experiment()