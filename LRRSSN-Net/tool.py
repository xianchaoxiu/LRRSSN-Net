
import torch
import torch.optim as optim
import scipy.io as sio
import numpy as np
import random, os
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 1. 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_tsne(
    features,
    labels,
    title="t-SNE Visualization",
    save_path=None,
    save_paths=None,
    class_colors=None,
    show_colorbar=True,
    show_legend=True,
    show_axis_ticks=True,
    show_plot=True,
):
    print(f"正在生成 {title}...")
    
    # 使用更简洁的初始化，增加 init='pca' 能让结果更稳定
    from sklearn.manifold import TSNE
    tsne = TSNE(
        n_components=2, 
        perplexity=30, 
        random_state=42, 
        init='pca', 
        learning_rate='auto'
    )
    
    # 2. 降维计算
    low_dim_embs = tsne.fit_transform(features)
    
    # 3. 绘图
    plt.figure(figsize=(10, 8))

    # 支持自定义类别颜色：
    # - dict: {label: color}
    # - list/tuple: 按排序后的类别顺序依次分配
    unique_labels = np.unique(labels)
    if class_colors is not None:
        if isinstance(class_colors, dict):
            label_to_color = {lab: class_colors.get(lab, 'gray') for lab in unique_labels}
        elif isinstance(class_colors, (list, tuple)):
            if len(class_colors) < len(unique_labels):
                raise ValueError(
                    f"class_colors 长度不足: 需要至少 {len(unique_labels)} 种颜色, "
                    f"当前仅有 {len(class_colors)} 种。"
                )
            label_to_color = {lab: class_colors[i] for i, lab in enumerate(unique_labels)}
        else:
            raise TypeError("class_colors 必须是 dict 或 list/tuple。")

        point_colors = [label_to_color[lab] for lab in labels]
        plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1],
                    c=point_colors, s=10, alpha=0.7)

        # 自定义颜色模式下用图例展示类别颜色
        handles = []
        for lab in unique_labels:
            handles.append(
                plt.Line2D([0], [0], marker='o', linestyle='',
                           markerfacecolor=label_to_color[lab],
                           markeredgecolor='none', markersize=6,
                           label=str(lab))
            )
        if show_legend:
            plt.legend(handles=handles, title='Class', bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        scatter = plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1],
                              c=labels, cmap='jet', s=10, alpha=0.7)
        if show_colorbar:
            plt.colorbar(scatter)

    if not show_axis_ticks:
        plt.xticks([])
        plt.yticks([])
    plt.title(title)
    
    targets = []
    if save_path:
        targets.append(save_path)
    if save_paths:
        targets.extend(save_paths)

    seen = set()
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        plt.savefig(target)
        print(f"图像已保存至: {target}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

# 评价指标
def align_cluster_labels(y_true, y_pred):
    y_true, y_pred = y_true.astype(np.int64), y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    label_mapping = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
    y_pred_aligned = np.array(
        [label_mapping.get(int(label), int(label)) for label in y_pred],
        dtype=np.int64,
    )
    acc = w[row_ind, col_ind].sum() * 1.0 / y_pred.size
    return acc, y_pred_aligned, label_mapping


def calculate_acc(y_true, y_pred):
    acc, _, _ = align_cluster_labels(y_true, y_pred)
    return acc

def evaluate_clustering(
    Z_tensor,
    labels_true,
    n_clusters,
    show_plot=False,
    return_details=False,
):
    """
    Z_tensor: 模型输出的 Z (Tensor)
    labels_true: 真实标签 (Numpy)
    n_clusters: 聚类数
    show_plot: 是否绘图开关
    """
    # 1. 转换数据
    Z_np = Z_tensor.cpu().detach().numpy()
    W = (np.abs(Z_np) + np.abs(Z_np.T)) / 2
    
    # 2. 控制绘图逻辑
    if show_plot:
        plt.figure(figsize=(6, 5))
        plt.imshow(W, cmap='jet')
        plt.title("Learned Affinity Matrix W")
        plt.colorbar()
        plt.show()
    
    # 3. 执行谱聚类
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                            assign_labels='kmeans', n_init=5, n_jobs=1, random_state=42)
    labels_pred = sc.fit_predict(W)
    
    # 4. 计算指标 (加入 ARI)
    acc, labels_pred_aligned, label_mapping = align_cluster_labels(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    
    results = {"ACC": acc, "NMI": nmi, "ARI": ari}
    if return_details:
        results["labels_pred"] = labels_pred
        results["labels_pred_aligned"] = labels_pred_aligned
        results["label_mapping"] = label_mapping
        results["affinity_matrix"] = W
    return results

# 可视化每一阶段Z
def visualize_stage_evolution(all_Z, save_dir="./stage_vis_Z"):
    """
    可视化并保存每一层 Stage 的 Z 矩阵热力图。
    all_Z: 包含每一层 Z tensor 的列表
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    num_stages = len(all_Z)
    print(f"\n开始生成 {num_stages} 个 Stage 的 Z 矩阵可视化...")

    # 设置统一的色彩量级，便于跨层对比
    # 获取最后一层 Z 的最大值作为参考，或者手动设定一个合理值 (如 0.5-1.0)
    with torch.no_grad():
        max_val = torch.max(torch.abs(all_Z[-1])).item() * 0.8 
        # 如果最后一层特别稀疏导致max_val太小，给一个底限
        max_val = max(max_val, 0.1) 

    for i, Z_tensor in enumerate(all_Z):
        # 1. 数据准备：转为 numpy，取绝对值
        Z_np = torch.abs(Z_tensor).detach().cpu().numpy()
        
        # 2. 对称化：构建谱聚类实际使用的亲和矩阵 W = (|Z| + |Z|^T) / 2
        W_np = (Z_np + Z_np.T) / 2
        
        # 3. 绘图
        plt.figure(figsize=(8, 8))
        # 使用 'magma' 或 'inferno' 色图，黑色背景更能凸显稀疏结构
        v_max = np.percentile(W_np, 99.9) # 选取最亮的 0.1% 作为参考
        plt.imshow(W_np, cmap='magma', vmin=0, vmax=v_max)
        plt.colorbar(label='Affinity Strength')
        
        title = f"Stage {i+1:02d}/{num_stages} Affinity Matrix Z\n(Symmetrized |Z|)"
        plt.title(title, fontsize=14)
        plt.xlabel("Sample Index")
        plt.ylabel("Sample Index")
        
        # 4. 保存路径
        save_path = os.path.join(save_dir, f"stage_{i+1:02d}_Z_heatmap.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close() # 关闭图像释放内存
        print(f"  - 已保存: {save_path}")
        
    print("所有 Stage 可视化完成。\n")


def plot_eigengap(Z, k_target=40, num_eigvals=60):
    """
    提取并绘制拉普拉斯矩阵的前 num_eigvals 个特征值
    Z: 模型输出的系数矩阵 (n x n)
    k_target: 目标簇数 (ORL数据集为 40)
    """
    with torch.no_grad():
        # 1. 构造对称亲和矩阵
        W = (torch.abs(Z) + torch.abs(Z.t())) / 2
        
        # 2. 计算度矩阵和拉普拉斯矩阵
        D = torch.diag(torch.sum(W, dim=1))
        L = D - W
        
        # 3. 计算特征值 (升序排列)
        # 使用 eigvalsh 针对对称矩阵进行稳定计算
        eigvals = torch.linalg.eigvalsh(L)
        
        # 4. 提取前 60 个特征值并转为 numpy
        top_eigvals = eigvals[:num_eigvals].cpu().numpy()

    # 5. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_eigvals + 1), top_eigvals, 'o-', color='blue', linewidth=2, markersize=5, label='Eigenvalues')
    
    # 标注目标簇数 k 的位置
    plt.axvline(x=k_target, color='red', linestyle='--', label=f'k={k_target} (Target Clusters)')
    
    # 设置图表属性
    plt.title("Laplacian Eigenvalues Distribution (Eigengap Analysis)", fontsize=14)
    plt.xlabel("Index of Eigenvalues", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # 重点观察区
    gap = top_eigvals[k_target] - top_eigvals[k_target-1]
    plt.annotate(f'Eigengap: {gap:.4f}', 
                 xy=(k_target, top_eigvals[k_target]), 
                 xytext=(k_target+5, top_eigvals[k_target]+0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.show()

# 在你的评估代码中使用：
# plot_eigengap(final_Z, k_target=40)