import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

def preprocess_point_cloud(pcd, voxel_size=0.005):
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.normalize_normals()
    return pcd

def estimate_curvature(pcd, radius=0.01):
    points = np.asarray(pcd.points)
    tree = KDTree(points)
    curvature = np.zeros(len(points))

    for i, point in enumerate(points):
        idx = tree.query_radius([point], r=radius)[0]
        if len(idx) < 5:
            curvature[i] = 1.0
            continue
        neighbors = points[idx]
        cov = np.cov((neighbors - point).T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)
        curvature[i] = eigvals[0] / eigvals.sum()
    return curvature

def vacuum_grasp_scores(normals, curvature):
    # 法线朝向 z+ 且曲率小的地方更适合吸盘
    normals = np.asarray(normals)
    y_alignment = np.clip(normals[:, 1], 0, 1)  # 只考虑朝上的点
    flatness = 1 - np.clip(curvature, 0, 1)
    scores = y_alignment * flatness
    return scores

def gripper_grasp_scores(pcd, normals, curvature, gripper_width=0.01, angle_threshold=0.9, radius=0.01):
    points = np.asarray(pcd.points)
    normals = np.asarray(normals)
    tree = KDTree(points)
    scores = np.zeros(len(points))

    for i, (pi, ni) in enumerate(zip(points, normals)):
        idx = tree.query_radius([pi], r=radius)[0]
        for j in idx:
            if i == j:
                continue
            pj, nj = points[j], normals[j]
            vec = pj - pi
            dist = np.linalg.norm(vec)
            if dist < 1e-3 or abs(dist - gripper_width) > 0.02:
                continue
            # 判断法线是否对向
            dot = np.dot(ni, nj)
            if dot > -angle_threshold:
                continue
            # 位置合适 + 对向法线
            score = (1 - abs(dot)) * np.exp(-((dist - gripper_width) ** 2) / (2 * 0.01**2))
            scores[i] = max(scores[i], score)
            scores[j] = max(scores[j], score)
    return scores

def visualize_scores(pcd, scores, title="Score Visualization"):
    cmap = plt.get_cmap("jet")
    norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-6)
    colors = cmap(norm_scores)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"Showing: {title}")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # 加载点云
    pcd = o3d.io.read_point_cloud("D:\\Codecouldcode\\099.MA_Hanyu\\Object\\v1.1\\object_04.pcd")  # 替换为你的文件路径
    pcd = preprocess_point_cloud(pcd)
    curvature = estimate_curvature(pcd)
    normals = pcd.normals

    # 吸盘评分
    vacuum_scores = vacuum_grasp_scores(normals, curvature)
    visualize_scores(pcd, vacuum_scores, "Vacuum Grasp Score")

    # 夹爪评分
    gripper_scores = gripper_grasp_scores(pcd, normals, curvature)
    visualize_scores(pcd, gripper_scores, "Gripper Grasp Score")
