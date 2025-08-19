
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


scene_version = "v1.1"
obj_num = "4"

path_obj = "D:\\Codecouldcode\\099.MA_Hanyu\\Data_Example\\model_"+scene_version+".ply"
path_pcd = "D:\\Codecouldcode\\099.MA_Hanyu\\scene_"+scene_version+".pcd"
save_dir = "D:\\Codecouldcode\\099.MA_Hanyu\\Object\\"+scene_version+"\\"

# path_exp = "D:\\Codecouldcode\\099.MA_Hanyu\\Object\\"+scene_version+"\\"+"object_0"+obj_num+".pcd"
path_exp = "D://Codecouldcode//099.MA_Hanyu//YCB_Dataset//019_pitcher_base.ply"

# 创建坐标轴
# size: 坐标轴长度，origin: 坐标轴原点
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

#夹爪宽度
gripper_width = 0.05


# mesh_scene = o3d.io.read_triangle_mesh(path_obj)
# o3d.visualization.draw_geometries([mesh_scene])

# mesh_scene.compute_vertex_normals()
# pcd_scene = mesh_scene.sample_points_poisson_disk(number_of_points=500000)

# o3d.visualization.draw_geometries([pcd_scene])
# o3d.io.write_point_cloud(path_pcd, pcd_scene)


pcd_scene = o3d.io.read_point_cloud(path_pcd)


#todo
# RANSAC 平面分割
plane_model, inliers = pcd_scene.segment_plane(distance_threshold=0.001,ransac_n=4,num_iterations=5000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

#计算桌面法向量
plane_normal = np.array([a, b, c])
plane_normal = plane_normal / np.linalg.norm(plane_normal)
print("桌面法向量:", plane_normal)

# 提取平面上的点
plane_cloud = pcd_scene.select_by_index(inliers)
plane_cloud.paint_uniform_color([1.0, 0, 0])  # 红色

# 提取剩余非平面点
rest_cloud = pcd_scene.select_by_index(inliers, invert=True)

# 可视化
# o3d.visualization.draw_geometries([plane_cloud, rest_cloud,axis])

#todo
#筛选平面上方点
points = np.asarray(rest_cloud.points)
# 计算每个点与平面的关系（大于0表示上方）
distances = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
# 设置容忍误差，防止浮点误差误杀平面上的点
threshold = 0 #1e-4
indices_above_plane = np.where(distances > threshold)[0]
# 选出平面上方的点
pcd_above = rest_cloud.select_by_index(indices_above_plane)


#todo
# 进行 DBSCAN 聚类
labels = np.array(pcd_above.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
max_label = labels.max()
num_clusters = max_label+1
print(f"Found {num_clusters} clusters.")

# 可视化每个聚类

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # 将噪声设置为黑色
pcd_above.colors = o3d.utility.Vector3dVector(colors[:, :3])

# o3d.visualization.draw_geometries([pcd_above,axis])




# #todo
# # 单独保存每个物体
# for i in range(num_clusters):
#     indices = np.where(labels == i)[0]
#     obj = pcd_above.select_by_index(indices)
#     save_path = os.path.join(save_dir, f"object_{i:02d}.pcd")
#     #创建父级目录
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     o3d.io.write_point_cloud(save_path, obj)
#     print(f"Saved: {save_path}")

#     # 可视化单个物体（可选）
#     o3d.visualization.draw_geometries([obj])


#todo
pcd_example = o3d.io.read_point_cloud(path_exp)
# o3d.visualization.draw_geometries([pcd_example])


points = np.asarray(pcd_example.points)
# Step 1: 计算质心
centroid = points.mean(axis=0)
print("质心位置：", centroid)
# 创建一个小球体标记质心
centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)  # 半径可以根据点云大小调整
centroid_sphere.translate(centroid)  # 移动到质心位置
centroid_sphere.paint_uniform_color([0, 1, 0])  # 绿色标记

# Step 2: 做 PCA（主成分分析）
# 去中心化
points_centered = points - centroid
# 计算协方差矩阵
cov = np.cov(points_centered.T)
# 计算特征值与特征向量
eig_vals, eig_vecs = np.linalg.eig(cov)

#判断特征值是否相近
eig_vals_sorted = np.sort(eig_vals)[::-1]
if (eig_vals_sorted[0] - eig_vals_sorted[1]) / eig_vals_sorted[0] < 0.05:
    print("⚠ 主轴不明确，可能是对称物体（如正方体）")
    # Step 2: 找到物体在桌面法向量方向上的投影范围（即“高度”）
    points = np.asarray(pcd_example.points)
    heights = points @ plane_normal  # 点云在法向量方向上的投影
    min_h, max_h = np.min(heights), np.max(heights)
    height_vector = (max_h - min_h) * plane_normal
    print("物体高度向量:", height_vector)
    # Step 3: 主轴定义为“从底部朝上的方向”
    main_axis = plane_normal  # 可以选择 plane_normal 或 -plane_normal
    print("定义主轴为:", main_axis)
else:


    # 按特征值大小排序（主轴 = 最大特征值对应的特征向量）
    sorted_indices = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, sorted_indices]
    eig_vals = eig_vals[sorted_indices]

    # 主轴方向
    main_axis = eig_vecs[:, 0]
    print("主轴向量（最大方差方向）：", main_axis)

# 可选：把主轴画出来（可视化）
axis_line = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector([
        centroid,
        centroid + main_axis * 0.1  # 线段长度自行调节
    ]),
    lines=o3d.utility.Vector2iVector([[0, 1]])
)
axis_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色

# 显示物体与主轴线
# o3d.visualization.draw_geometries([pcd_example, axis_line,centroid_sphere])


#todo 做伽马平面&筛选点云
def extract_slice_by_axis(pcd, main_axis, centroid, distance_thresh=0.007):
    points = np.asarray(pcd.points)

    # 确保主轴为单位向量
    main_axis = main_axis / np.linalg.norm(main_axis)

    # 点到平面的有符号距离： (P - C) · n
    vectors = points - centroid
    signed_distances = np.dot(vectors, main_axis)

    # 筛选绝对距离小于阈值的点（7mm）
    mask = np.abs(signed_distances) < distance_thresh
    sliced_points = points[mask]

    # 构建新的点云对象
    slice_pcd = o3d.geometry.PointCloud()
    slice_pcd.points = o3d.utility.Vector3dVector(sliced_points)

    return slice_pcd

# 假设你已经有：
# - object_pcd: 原始物体点云
# - main_axis: 通过PCA计算得到的主轴向量
# - centroid: 质心坐标

slice_pcd = extract_slice_by_axis(pcd_example, main_axis, centroid, 0.007)
slice_pcd.paint_uniform_color([0, 1, 0])  # 绿色标记

# 可视化
# o3d.visualization.draw_geometries([slice_pcd,pcd_example,axis_line,centroid_sphere])
o3d.visualization.draw_geometries([slice_pcd,pcd_example,axis_line,centroid_sphere,axis])


#todo 寻找候选点区域

def get_p1_p2_from_slice(slice_pcd, main_axis, plane_normal):
    points = np.asarray(slice_pcd.points)

    # 归一化向量
    main_axis = main_axis / np.linalg.norm(main_axis)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # 计算夹角余弦（取绝对值，关心角度大小）
    parallel_to_plane = np.abs(np.dot(main_axis, plane_normal)) < 0.5  # 越小越平行
    closer_to_x = np.abs(np.dot(main_axis, [1, 0, 0])) > np.abs(np.dot(main_axis, [0, 0, 1]))

    if parallel_to_plane and closer_to_x:
        # 主轴接近平面且靠近 X 轴 —— 取 Z 最大/最小
        z_vals = points[:, 2]
        idx_max = np.argmax(z_vals)
        idx_min = np.argmin(z_vals)
        p1, p2 = points[idx_max], points[idx_min]
    else:
        # 主轴更垂直，靠近 Z 轴 —— 取 X 最大/最小
        x_vals = points[:, 0]
        idx_max = np.argmax(x_vals)
        idx_min = np.argmin(x_vals)
        p1, p2 = points[idx_max], points[idx_min]

    return p1, p2


p1, p2 = get_p1_p2_from_slice(slice_pcd, main_axis, plane_normal)
print("关键点坐标：")
print("p1:", p1)
print("p2:", p2)
p1_mark = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)  # 半径可以根据点云大小调整
p2_mark = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)  # 半径可以根据点云大小调整
p1_mark.translate(p1)  # 移动到质心位置
p2_mark.translate(p2)  # 移动到质心位置
p1_mark.paint_uniform_color([1, 0, 0])  # 红色标记
p2_mark.paint_uniform_color([1, 0, 0])  # 红色标记

o3d.visualization.draw_geometries([slice_pcd,pcd_example,axis_line,centroid_sphere,axis,p1_mark,p2_mark])

#todo 计算候选区域
def find_points_within_sphere(pcd, center, radius):

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    [_, idxs, _] = kdtree.search_radius_vector_3d(center, radius)
    return idxs

#用p1 p2 距离近似等于物体的宽度
width_obj = np.linalg.norm(p2 - p1)
radius = 2*gripper_width
#如果物体宽度过小
while width_obj <= 2*radius:
    radius = width_obj*0.9/2

# 获取 Q1 和 Q2 的索引
Q1_idxs = find_points_within_sphere(slice_pcd, p1, radius)
Q2_idxs = find_points_within_sphere(slice_pcd, p2, radius)

# 提取点云
points = np.asarray(slice_pcd.points)

Q1_orig = o3d.geometry.PointCloud()
Q1_orig.points = o3d.utility.Vector3dVector(points[Q1_idxs])
Q1_orig.paint_uniform_color([0, 0, 1]) #黄色

Q2_orig = o3d.geometry.PointCloud()
Q2_orig.points = o3d.utility.Vector3dVector(points[Q2_idxs])
Q2_orig.paint_uniform_color([0, 0, 1])  # 黄色

# 可视化
o3d.visualization.draw_geometries([slice_pcd.translate((0,0,0.001)),pcd_example.translate((0,0,-0.001)),axis_line,centroid_sphere,axis,p1_mark,p2_mark,Q1_orig,Q2_orig])

#todo 候选点ranking


#体素采样
voxel_radius = 0.05*gripper_width
Q1_voxel = Q1_orig.voxel_down_sample(voxel_size=voxel_radius)
Q2_voxel = Q2_orig.voxel_down_sample(voxel_size=voxel_radius)

'''
# dist函数
'''
def compute_dist_scores(points, principal_axis, centroid):
    """
    points: (N, 3) numpy array of Q1 or Q2 point cloud
    principal_axis: (3,) unit vector, main axis of the object
    centroid: (3,) centroid of the object
    return: (N,) array of normalized scores in [0.0, 1.0]
    """
    n = principal_axis / np.linalg.norm(principal_axis)  # 确保单位向量
    offset = -np.dot(n, centroid)  # 平面偏移量 两点作向量，与法向量点成后，只剩竖直分量，可得距离。此处n其实是原点与质心的差

    # 原始距离值
    raw_dists = np.abs(np.dot(points, n) + offset)
    # raw_dists = np.abs(np.dot(points, n))

    # 归一化为 [0, 1]
    min_dist = np.min(raw_dists)
    max_dist = np.max(raw_dists)

    if max_dist == min_dist:
        return np.zeros_like(raw_dists)
    else:
        return (raw_dists - min_dist) / (max_dist - min_dist)

# points_q1 = np.asarray(Q1_voxel.points)
# scores_q1 = compute_dist_scores(points_q1, main_axis, centroid)
# points_q2 = np.asarray(Q2_voxel.points)
# scores_q2 = compute_dist_scores(points_q2, main_axis, centroid)

# 可视化颜色映射（红色 = 得分高，蓝色 = 得分低）
# colors = plt.get_cmap("coolwarm")(scores_q1)[:, :3]  # RGB from colormap
# Q1_voxel.colors = o3d.utility.Vector3dVector(colors)
# colors = plt.get_cmap("coolwarm")(scores_q2)[:, :3]  # RGB from colormap
# Q2_voxel.colors = o3d.utility.Vector3dVector(colors)

# o3d.visualization.draw_geometries([Q1_voxel,Q2_voxel])

'''
# 曲率
'''


def estimate_point_curvature(pcd, radius=0.05):
    """    
    
    对点云中的每个点计算：
    - 曲率值 curvature
    - 最小特征值方向（曲率主方向）normal_vec
    
    参数:
        pcd (open3d.geometry.PointCloud): 输入点云
        radius (float): 邻域搜索半径，用于PCA计算
    返回:
        curvatures (np.ndarray): shape=(N,) 的曲率数组，已归一化到 [0, 1]
        curvature_dirs = []
    """

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    curvatures = []
    curvature_dirs = []

    for i in range(len(points)):
        [_, idxs, _] = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        if len(idxs) < 3:
            curvatures.append(0.0)
            curvature_dirs.append(np.array([0.0, 0.0, 0.0]))
            continue
        neighbors = points[idxs, :]
        covariance = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        eigvals = np.sort(eigvals)
        lambda0 = eigvals[0]
        curvature = lambda0 / np.sum(eigvals)
        curvatures.append(curvature)

        # 对应的特征向量（曲率主方向）是 eigvecs 的第0列
        curvature_dirs.append(eigvecs[:, 0])  # eigenvectors是列向量形式

    curvatures = np.array(curvatures)
    curvature_dirs = np.array(curvature_dirs)  # shape = (N, 3)
    # 归一化到 [0, 1]
    min_c, max_c = curvatures.min(), curvatures.max()
    if max_c > min_c:
        curvatures = (curvatures - min_c) / (max_c - min_c)
    else:
        curvatures = np.zeros_like(curvatures)
    
    return curvatures,curvature_dirs


# 假设 pcd 是一个 open3d.geometry.PointCloud 点云对象
# curvatures1,curvature_dirs1 = estimate_point_curvature(Q1_voxel, radius=0.014)
# curvatures2,curvature_dirs2 = estimate_point_curvature(Q2_voxel, radius=0.014)


# # 将曲率映射为颜色（蓝->红）
# colors = plt.get_cmap("jet")(curvatures1)[:, :3]
# Q1_voxel.colors = o3d.utility.Vector3dVector(colors)
# colors = plt.get_cmap("jet")(curvatures2)[:, :3]
# Q2_voxel.colors = o3d.utility.Vector3dVector(colors)

# 可视化
# o3d.visualization.draw_geometries([Q1_voxel,slice_pcd.translate((0,0,0.001)),pcd_example.translate((0,0,-0.001)),axis_line,centroid_sphere,axis,p1_mark,p2_mark])
# o3d.visualization.draw_geometries([Q1_voxel,Q2_voxel])

'''
# 夹角
'''

def compute_angle_cos(v1, v2):
    """计算两个向量的夹角余弦（弧度），返回余弦值"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    # return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return abs(np.dot(v1_u, v2_u))


'''
# ranking
'''

def rank_pair(q1, q2, dist1, dist2, lambda1, lambda2, n1, n2, w1=1.0, w2=1.0):
    # === r1: 位置评分 ===
    w = q2 - q1
    cos_beta = compute_angle_cos(w, main_axis)
    r1 = (1 - dist1) + (1 - dist2) - (cos_beta - 0.2) * 10.0 

    # === r2: 曲率评分 ===
    cos_alpha1 = compute_angle_cos(n1, w)
    cos_alpha2 = compute_angle_cos(n2, w)
    r2 = (1 - lambda1) + (1 - lambda2) + cos_alpha1 + cos_alpha2 - abs(cos_alpha1 - cos_alpha2)
    return w1 * r1 + w2 * r2

best_score = -np.inf
p1_idx, p2_idx = None, None

q1_voxel_points = np.asarray(Q1_voxel.points)
q2_voxel_points = np.asarray(Q2_voxel.points)
dist1 = compute_dist_scores(q1_voxel_points,main_axis,centroid)
dist2 = compute_dist_scores(q2_voxel_points,main_axis,centroid)
lambda1,normal1 = estimate_point_curvature(Q1_voxel, radius=0.014)
lambda2,normal2 = estimate_point_curvature(Q2_voxel, radius=0.014)


for i in range(len(q1_voxel_points)):
    for j in range(len(q2_voxel_points)):
        q1_i = q1_voxel_points[i]
        q2_j = q2_voxel_points[j]
        lambda1_i = lambda1[i]
        lambda2_j = lambda2[j]
        n1_i = normal1[i]
        n2_j = normal2[j]

        score = rank_pair(q1_i, q2_j, dist1[i], dist2[j], lambda1_i, lambda2_j, n1_i, n2_j)
        if score >= best_score:
            best_score = score
            print(f'best_score={best_score}')
            p1_idx, p2_idx = i,j
        elif score == 10.0:
            print(f'222best_score={best_score}')

p1_fit_mark = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)  # 半径可以根据点云大小调整
p1_fit_mark.translate(Q1_voxel.points[p1_idx])  # 移动p1最佳点
p1_fit_mark.paint_uniform_color([0, 0, 1])  # 蓝色标记
p2_fit_mark = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)  # 半径可以根据点云大小调整
p2_fit_mark.translate(Q2_voxel.points[p2_idx])  # 移动p2最佳点
p2_fit_mark.paint_uniform_color([0, 0, 1])  # 蓝色标记


o3d.visualization.draw_geometries([Q1_voxel,Q2_voxel,slice_pcd.translate((0,0,0.001)),pcd_example.translate((0,0,-0.001)),axis_line,centroid_sphere,axis,p1_fit_mark,p2_fit_mark])

