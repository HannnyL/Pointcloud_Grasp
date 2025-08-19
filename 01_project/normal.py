

import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter

# ---------- 参数设置 ----------
point_cloud_path = r"D:\Codecouldcode\099.MA_Hanyu\Object\cube_ring_sampled.pcd"  # ← 修改为你的点云文件
normal_radius = 0.05
curvature_radius = 0.05
dbscan_eps = 0.1
dbscan_min_samples = 20
local_k = 20  # 用于局部一致性纠正的邻域点数

# ---------- 加载点云 ----------
pcd = o3d.io.read_point_cloud(point_cloud_path)
print(f"Loaded point cloud with {len(pcd.points)} points")

# ---------- 步骤 1：估计法向 ----------
print("Estimating normals...")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
)

# ---------- 步骤 2：聚类法向，找到主方向 ----------
print("Clustering normals...")
normals = np.asarray(pcd.normals)
normals_unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)

clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='cosine')
labels = clustering.fit_predict(normals_unit)

label_counts = Counter(labels[labels >= 0])
most_common_label = label_counts.most_common(1)[0][0]
main_cluster_indices = np.where(labels == most_common_label)[0]
main_direction = normals[main_cluster_indices].mean(axis=0)
main_direction /= np.linalg.norm(main_direction)
print(f"Main normal direction (cluster {most_common_label}): {main_direction}")

# ---------- 步骤 3：统一所有法向朝主方向 ----------
print("Flipping normals toward main direction...")
flipped_normals = []
for i in range(len(normals)):
    n = normals[i]
    if np.dot(n, main_direction) < 0:
        n = -n
    flipped_normals.append(n)
pcd.normals = o3d.utility.Vector3dVector(np.array(flipped_normals))

# ---------- 步骤 4：局部一致性修正 ----------
def fix_local_inconsistent_normals(pcd, neighbor_k=20):
    print("Fixing local inconsistent normals...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = np.asarray(pcd.normals)
    fixed_normals = normals.copy()

    for i in range(len(normals)):
        _, idx, _ = pcd_tree.search_knn_vector_3d(pcd.points[i], neighbor_k)
        neigh_normals = normals[idx]
        dot_products = np.dot(neigh_normals, normals[i])
        flip_ratio = np.sum(dot_products < 0) / len(dot_products)
        if flip_ratio > 0.5:
            fixed_normals[i] = -normals[i]
    pcd.normals = o3d.utility.Vector3dVector(fixed_normals)

fix_local_inconsistent_normals(pcd, neighbor_k=local_k)

# ---------- 可选（进一步优化连续性） ----------
print("Optimizing consistency using MST...")
pcd.orient_normals_consistent_tangent_plane(k=local_k)

# ---------- 可视化 ----------
print("Showing result...")
o3d.visualization.draw_geometries(
    [pcd],
    point_show_normal=True,
    window_name="法向方向统一结果",
    width=800, height=600
)
