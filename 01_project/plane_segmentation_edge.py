import open3d as o3d
import numpy as np
import random
import copy
from math import acos, degrees
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from collections import Counter
import math

import alphashape
from shapely.geometry import Point, MultiPoint,Polygon
import matplotlib.pyplot as plt
from scipy.spatial import KDTree,cKDTree
import cv2
import itertools
# -------------------- Step 0: 定义函数 ----------------
class parallel_gripper:
    def __init__(self):
        space = 0.005
        a_pg = 0.01 # Finger width
        w_pg = 0.3*space # Internal Safespace Finger width 
        v_pg = space # External Safespace Finger width 
        f_pg = 0.10 # Distance gripper open
        g_pg = 0.02 # Distance gripper close
        h_pg = 0.12 # Gripper base bottom width
        k_pg = space # Safespace Gripper base bottom width 
        q_pg = 0.08 # Gripper base top width
        r_pg = space # Safespace Gripper base top width

        # b_pg = 0.01 # TCP to Finger length end
        c_pg = 0.04 # TCP to (Safety space of Gripper)length end
        d_pg = space # Safespace Gripper length
        x_pg = space # Safespace Gripper end to rubber
        n_pg = d_pg + c_pg + x_pg # Finger length
        t_pg = 0.065 # Gripper base bottom length
        u_pg = 0.05 # Gripper base top length
        j_pg = c_pg + d_pg + t_pg + u_pg # Gripper length (TCP to Robot)
        s_pg = j_pg + x_pg # Total gripper length

        e_pg = 0.04 # Finger depth
        i_pg = space # Safespace finger depth
        l_pg = 0.06 # Gripper base bottom depth
        m_pg = space # Safespace gripper base bottom depth
        o_pg = 0.07 # Gripper base top  depth
        p_pg = space # Safespace gripper base top depth

        y_pg = l_pg/2 + m_pg if (l_pg/2 + m_pg) >  (o_pg/2 + p_pg) else (o_pg/2 + p_pg) # Gripper Bounding box depth

space = 0.005
a_pg = 0.01 # Finger width
w_pg = 0.3*space # Internal Safespace Finger width 
v_pg = space # External Safespace Finger width 
f_pg = 0.10 # Distance gripper open
g_pg = 0.02 # Distance gripper close
h_pg = 0.12 # Gripper base bottom width
k_pg = space # Safespace Gripper base bottom width 
q_pg = 0.08 # Gripper base top width
r_pg = space # Safespace Gripper base top width

# b_pg = 0.01 # TCP to Finger length end
c_pg = 0.04 # TCP to (Safety space of Gripper)length end
d_pg = space # Safespace Gripper length
x_pg = space # Safespace Gripper end to rubber
n_pg = d_pg + c_pg + x_pg # Finger length
t_pg = 0.065 # Gripper base bottom length
u_pg = 0.05 # Gripper base top length
j_pg = c_pg + d_pg + t_pg + u_pg # Gripper length (TCP to Robot)
s_pg = j_pg + x_pg # Total gripper length

e_pg = 0.04 # Finger depth
i_pg = space # Safespace finger depth
l_pg = 0.12 # Gripper base bottom depth
m_pg = space # Safespace gripper base bottom depth
o_pg = 0.07 # Gripper base top  depth
p_pg = space # Safespace gripper base top depth

y_pg = l_pg/2 + m_pg if (l_pg/2 + m_pg) >  (o_pg/2 + p_pg) else (o_pg/2 + p_pg) # Gripper Bounding box depth

radius_robot = 0.1
lenth_robot = 0.2
angle_robot = math.radians(45)



# **************************** Step 1: 读取点云+外法线估计 ****************************
point_cloud_path = r"D:\Codecouldcode\099.MA_Hanyu\Object\Unregular_box_sampled.pcd"  # ← 修改为你的点云文件
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
    window_name="Estimated external normal",
    width=800, height=600
)
# ************************
original_points = np.asarray(pcd.points)
original_indices = np.arange(len(original_points))
# ---------------- Step 2: 初始化 ----------------
plane_indices_list = []   # 存放每个平面的原始索引
plane_colors = []         # 用于可视化
plane_models = []         # 平面模型
plane_normals = []        # 平面法向量

min_remaining_points = 100
min_points_per_plane = 50      # 小平面过滤阈值

distance_threshold = 0.001       # 拟合误差
max_planes = 100              # 最多检测几个平面（可调大）

rest_pcd = copy.deepcopy(pcd)

# ---------------- Step 3: 提取多个平面 ----------------

def correct_normal_direction_by_density(pcd, plane_indices, plane_normal):
        
        all_normals = np.asarray(pcd.normals)
        outer_avg = all_normals[plane_indices].mean(axis=0)
        outer_avg /= np.linalg.norm(outer_avg)

        dot = np.dot(outer_avg, plane_normal)
        angle = np.arccos(np.clip(dot, -1.0, 1.0)) * 180 / np.pi
        print(f"average point normal vs fitted normal angle: {angle:.2f}°")
        
        # 与原始平面法向比一比，看需不需要翻转
        if angle > 90 :
            plane_normal = -plane_normal  # 让它朝外 
        
        return plane_normal

for i in range(max_planes):

    print(f"loop{i}")

    if len(rest_pcd.points) < min_remaining_points:
        print("Not enough points left to extract more planes.")
        break  # 剩余点太少，停止提取

    plane_model, inliers = rest_pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000)

    if len(inliers) < min_points_per_plane:
        print(f"Plane has only {len(inliers)} points, skipping.")
        break  # 太小，停止提取

    # 保存当前平面的点索引（是当前点云的 index，需要映射回原始点云）
    current_points = np.asarray(rest_pcd.points)
    current_indices = np.arange(len(current_points))
    original_idx = original_indices[inliers]
    plane_indices_list.append(original_idx)

    ##todo 去除被切穿点*****


# ***************************************************
    #保存平面模型与法线
    plane_models.append(plane_model) # 平面模型
    normal_vector = np.asarray(plane_model[0:3])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    normal_vector = correct_normal_direction_by_density(pcd, original_idx, normal_vector)
    plane_normals.append(normal_vector) # 平面法向量

    # 为当前平面着色（随机）
    color = [random.random(), random.random(), random.random()]
    plane_colors.append(color)

    # 剩余点云
    rest_pcd = rest_pcd.select_by_index(inliers, invert=True)
    original_indices = np.delete(original_indices, inliers)

# ---------------- Step 4: 可视化着色 ----------------
# 创建一个点云，给所有点上色
colored_pcd = copy.deepcopy(pcd)
colors = np.ones((len(original_points), 3)) * [0.5, 0.5, 0.5]  # 默认灰色

for indices, color in zip(plane_indices_list, plane_colors):
    colors[indices] = color  # 为每个平面的点赋色

colored_pcd.colors = o3d.utility.Vector3dVector(colors)

#创建法线显示
def create_normal_arrow(origin, normal, length=0.02, color=[1, 0, 0]):
    """
    创建一个法线箭头用于可视化
    origin: 起点（三维坐标）
    normal: 单位法向量
    length: 箭头长度
    color: RGB颜色
    """
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.001,
        cone_radius=0.002,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2
    )
    arrow.paint_uniform_color(color)

    # 构造旋转矩阵，使箭头从Z方向旋转到 normal
    z_axis = np.array([0, 0, 1])
    normal = normal / np.linalg.norm(normal)
    v = np.cross(z_axis, normal)
    c = np.dot(z_axis, normal)
    if np.linalg.norm(v) < 1e-6:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v)**2))

    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(origin)
    return arrow


arrow_list = []

for indices, normal in zip(plane_indices_list, plane_normals):
    pts = np.asarray(pcd.select_by_index(indices.tolist()).points)
    center = np.mean(pts, axis=0)

    arrow = create_normal_arrow(center, normal, length=0.015, color=[1, 0, 0])
    arrow_list.append(arrow)




# ---------------- 可选：输出平面索引列表 ----------------
for i, indices in enumerate(plane_indices_list):
    print(f"Plane {i}: {len(indices)} points, indices example: {indices[:5]}")
    # o3d.visualization.draw_geometries([pcd.select_by_index(indices.tolist())], window_name=f"Plane {i} points", width=800, height=600)


#显示法线和平面
o3d.visualization.draw_geometries([colored_pcd] + arrow_list, window_name="Plane segmentation result and external normal",point_show_normal=False)

####################################################################


def is_parallel(v1, v2, angle_thresh_deg=5):
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = degrees(acos(abs(cos_theta)))  # abs保证±方向一致也算平行
    return angle <= angle_thresh_deg

unclustered = set(range(len(plane_normals)))
parallel_groups = []

while unclustered:
    idx = unclustered.pop()
    ref_normal = plane_normals[idx]

    current_group = [idx]

    to_remove = []
    for other in unclustered:
        if is_parallel(ref_normal, plane_normals[other]):
            current_group.append(other)
            to_remove.append(other)

    for i in to_remove:
        unclustered.remove(i)

    parallel_groups.append(current_group)

#颜色

colored_pcd = copy.deepcopy(pcd)
colors = np.ones((len(pcd.points), 3)) * [0.5, 0.5, 0.5]  # 灰色默认背景

group_colors = [[random.random(), random.random(), random.random()] for _ in parallel_groups]

for group_idx, group in enumerate(parallel_groups):
    color = group_colors[group_idx]
    for plane_idx in group:
        point_indices = plane_indices_list[plane_idx]
        colors[point_indices] = color

colored_pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([colored_pcd], window_name="Planeclustering result and external normal")

###############################################平面配对

#todo: 平面配对
def is_opposite_direction(idx_i, idx_j):

    pn1 = np.asarray(plane_normals[idx_i])
    pn2 = np.asarray(plane_normals[idx_j])

    point_indices_i = plane_indices_list[idx_i]
    point_indices_j = plane_indices_list[idx_j]
    plane_points_i = np.asarray(pcd.select_by_index(point_indices_i).points)
    plane_points_j = np.asarray(pcd.select_by_index(point_indices_j).points)
    pc1 = np.mean(plane_points_i, axis=0)
    pc2 = np.mean(plane_points_j, axis=0)

    c1c2 = pc2 - pc1 #中心连线 1->2

    dot0 = np.dot(pn1, pn2)
    dot1 = np.dot(c1c2, pn1) #外法线时应为-
    dot2 = np.dot(c1c2, pn2) #外法线时应为+


    if dot0 < 0 and dot1 < 0 and dot2 > 0:  # 方向相同
        return True
    else:   
        return False



paired_planes = []  # 所有配对

for group in parallel_groups:
    n = len(group)
    for i in range(n):
        for j in range(i + 1, n):  # 防止 (i,j) 和 (j,i) 重复
            idx_i = group[i]
            idx_j = group[j]

            n1 = plane_normals[idx_i]
            n2 = plane_normals[idx_j]

            if is_opposite_direction(idx_i, idx_j):
                paired_planes.append((idx_i, idx_j))

#可视化
for count, (i, j) in enumerate(paired_planes):
    # 创建新的颜色数组，初始为灰色
    colors = np.ones((len(pcd.points), 3)) * [0.6, 0.6, 0.6]

    # 为当前配对指定颜色
    color = [random.random(), random.random(), random.random()]
    for idx in plane_indices_list[i]:
        colors[idx] = color
    for idx in plane_indices_list[j]:
        colors[idx] = color

    # 创建新点云并赋色
    paired_pcd = copy.deepcopy(pcd)
    paired_pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Show pair：{i} ↔ {j}")
    # o3d.visualization.draw_geometries([paired_pcd], window_name=f"Pair {count+1}: Plane {i} 和 Plane {j}",width=800,height=600,)

#---------------Find center plane---------------

iii=0

(mmm,nnn) = paired_planes[iii]
plane_i_points = np.asarray(pcd.select_by_index(plane_indices_list[mmm]).points)
plane_j_points = np.asarray(pcd.select_by_index(plane_indices_list[nnn]).points)
center_i = np.mean(plane_i_points, axis=0)
center_j = np.mean(plane_j_points, axis=0)

center_ij = (center_i + center_j) / 2

dist_i = abs(np.dot((center_ij - center_i),plane_normals[mmm]))
dist_j = abs(np.dot((center_ij - center_j),plane_normals[nnn]))

# project_i_dir = (center_ij - center_i) / np.linalg.norm(center_ij - center_i)
# project_j_dir = (center_ij - center_j) / np.linalg.norm(center_ij - center_j)

projected_i_points = plane_i_points - np.outer(dist_i,plane_normals[mmm])

projected_j_points = plane_j_points - np.outer(dist_j,plane_normals[nnn])

pcd_proj_i = o3d.geometry.PointCloud()
pcd_proj_i.points = o3d.utility.Vector3dVector(projected_i_points)
pcd_proj_i.paint_uniform_color([0, 1, 0])  # 绿色

pcd_proj_j = o3d.geometry.PointCloud()
pcd_proj_j.points = o3d.utility.Vector3dVector(projected_j_points)
pcd_proj_j.paint_uniform_color([1, 0, 0])  # 红色

pcd_orig_i = pcd.select_by_index(plane_indices_list[mmm])
pcd_orig_j = pcd.select_by_index(plane_indices_list[nnn])

pcd_orig_i.paint_uniform_color([0.7, 0.7, 0.7])  # 淡灰色
pcd_orig_j.paint_uniform_color([0.7, 0.7, 0.7])

o3d.visualization.draw_geometries([
    pcd,
    pcd_orig_i,
    pcd_orig_j,
    pcd_proj_i,
    pcd_proj_j
], window_name="Plane Projection", width=800, height=600)

#**************************** Plane 1: Project planes and find overlap region ****************************

def extract_overlap_region(proj_A, proj_B, threshold=0.001,remove = False):
    """
    从两个已投影的点云中提取重合区域，返回合并后的重合点云
    """
    # 构建 KDTree
    kdtree_B = o3d.geometry.KDTreeFlann(proj_B)
    kdtree_A = o3d.geometry.KDTreeFlann(proj_A)

    points_A = np.asarray(proj_A.points)
    points_B = np.asarray(proj_B.points)

    # A 中那些有邻近 B 点的
    matched_A = []
    dismatched_A = []
    for p in points_A:
        [_, idx, _] = kdtree_B.search_radius_vector_3d(p, threshold)
        if len(idx) > 0:
            matched_A.append(p)
        else:
            dismatched_A.append(p)

    if remove == True:
        pcd_remove_overlap = o3d.geometry.PointCloud()
        pcd_remove_overlap.points = o3d.utility.Vector3dVector(dismatched_A)
        pcd_remove_overlap.paint_uniform_color([1, 0, 0])

        return pcd_remove_overlap
    else:
        # B 中那些有邻近 A 点的
        matched_B = []
        for p in points_B:
            [_, idx, _] = kdtree_A.search_radius_vector_3d(p, threshold)
            if len(idx) > 0:
                matched_B.append(p)

        # 合并重合区域的点
        overlap_points = np.vstack([matched_A, matched_B])

        pcd_overlap = o3d.geometry.PointCloud()
        pcd_overlap.points = o3d.utility.Vector3dVector(overlap_points)
        pcd_overlap.paint_uniform_color([0, 1, 0])  # 绿色标记

        return pcd_overlap

overlap_pcd = extract_overlap_region(pcd_proj_i, pcd_proj_j, threshold=0.001)

# 可视化重合区域
# pcd_proj_i.paint_uniform_color([0, 1, 0])
# pcd_proj_j.paint_uniform_color([0, 0, 1])

o3d.visualization.draw_geometries([overlap_pcd,pcd_orig_i,pcd_orig_j],window_name="Pair of Planes and Their Overlap Region")

#**************************** Plane 2: Find points between planes ****************************
def project_points_to_plane(points, plane_point, plane_normal):
    v = points - plane_point
    d = np.dot(v, plane_normal)
    return points - np.outer(d, plane_normal)

def select_points_between_planes(pcd, center_i, center_j, plane_normal, margin=0.0005, include_planes=True):
    """
    筛选出完整点云中夹在两个平面之间的点
    """

    if isinstance(pcd, o3d.geometry.PointCloud):
        points = np.asarray(pcd.points)
    elif isinstance(pcd, np.ndarray):
        points = pcd
    # center_i = plane_i_pts.mean(axis=0)
    # center_j = plane_j_pts.mean(axis=0)

    # 平面间距离向量（方向需与法向一致）
    dist_vec = center_j - center_i
    dist_vec /= np.linalg.norm(dist_vec)
    
    # 投影每个点到法向上，得到相对两个平面的距离
    d_i = np.dot(points - center_i, plane_normal)
    d_j = np.dot(points - center_j, plane_normal)

    # 判断点是否在两个平面之间（允许一个 margin 容差）
    if include_planes:
        mask = (d_i * d_j <= 0) | (np.abs(d_i) <= margin) | (np.abs(d_j) <= margin)
    else:
        mask = (d_i * d_j < 0) & (np.abs(d_i) > margin) & (np.abs(d_j) > margin)

    points_between = points[mask]
    points_beside = points[~mask]
    return points_between,points_beside




# 1. 筛选出在两个平面之间的点
points_between_p2,points_beside = select_points_between_planes(pcd, center_i, center_j, plane_normals[mmm])

# 2. 投影到中间平面
projected_points_p2 = project_points_to_plane(points_between_p2, center_ij, plane_normals[mmm])

# 3. 创建 PointCloud 对象
proj_pcd_p2 = o3d.geometry.PointCloud()
proj_pcd_p2.points = o3d.utility.Vector3dVector(projected_points_p2)
proj_pcd_p2.paint_uniform_color([1, 0, 0])  # 红色


# 可视化所有内容
o3d.visualization.draw_geometries([
    pcd_orig_j, pcd_orig_i, proj_pcd_p2
    ,
],window_name="Pair of Planes and Projected Points Between Them")


#**************************** Plane 3: find outside(finger) collision area ****************************

# center_i_outside = center_i + (y_pg) * (plane_normals[mmm])
# center_j_outside = center_j + (y_pg) * (plane_normals[nnn])
center_i_p3 = center_i + (0.02) * (plane_normals[mmm])
center_j_p3 = center_j + (0.02) * (plane_normals[nnn])

# 1. 筛选出在两个平面之间的点
points_between_p3_i,points_beside = select_points_between_planes(points_beside, center_i, center_i_p3, plane_normals[mmm],0.001,False)
points_between_p3_j,points_beside = select_points_between_planes(points_beside, center_j, center_j_p3, plane_normals[nnn],0.001,False)
points_between_p3 = np.vstack((points_between_p3_i, points_between_p3_j))

# 2. 投影到中间平面
projected_points_p3 = project_points_to_plane(points_between_p3, center_ij, plane_normals[mmm])

# 3. 创建 PointCloud 对象
proj_pcd_p3 = o3d.geometry.PointCloud()
proj_pcd_p3.points = o3d.utility.Vector3dVector(projected_points_p3)
proj_pcd_p3.paint_uniform_color([0, 0, 1])  # 蓝色

#平面中心
# sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
# sphere1.paint_uniform_color([1, 1, 0])  
# sphere1.translate(center_i)

# sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
# sphere2.paint_uniform_color([1, 1, 0])  
# sphere2.translate(center_j)

# sphere3 = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
# sphere3.paint_uniform_color([0, 1, 1]) 
# sphere3.translate(center_i_outside)

# sphere4 = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
# sphere4.paint_uniform_color([1, 0, 1])  
# sphere4.translate(center_j_outside)

# aaaa = o3d.geometry.PointCloud()
# aaaa.points = o3d.utility.Vector3dVector(np.array(points_between_outside_i+0.0001))
# aaaa.paint_uniform_color([1, 0, 1])
# bbbb = o3d.geometry.PointCloud()
# bbbb.points = o3d.utility.Vector3dVector(np.array(points_between_outside_j+0.0001))
# bbbb.paint_uniform_color([0, 1, 1])
# o3d.visualization.draw_geometries([pcd,aaaa,bbbb])
# 可视化所有内容
# o3d.visualization.draw_geometries([candidate_TCP_pcd, proj_pcd_outside,sphere1,sphere2,sphere3,sphere4])
o3d.visualization.draw_geometries([overlap_pcd, proj_pcd_p3.translate([0,0,0.0001])],window_name="Initial TCP & Finger Collision Area (P3)")

##**************************** Plane 4: find beside collision area ****************************

projected_points_p4 = project_points_to_plane(points_beside, center_ij, plane_normals[mmm])

# 3. 创建 PointCloud 对象
proj_pcd_p4 = o3d.geometry.PointCloud()
proj_pcd_p4.points = o3d.utility.Vector3dVector(projected_points_p4)
proj_pcd_p4.paint_uniform_color([0, 1, 1])  # 蓝色

o3d.visualization.draw_geometries([overlap_pcd, proj_pcd_p3.translate([0,0,0.0001]), proj_pcd_p4.translate([0,0,-0.0001])],window_name="Initial TCP & Finger Collision Area & Robot Collision Area (P4)")

#**************************** P2: Find contours ****************************


#-----------------------chat-cv2------------------------

def extract_and_visualize_contour_segments_with_normals(pcd, scale=1500, approx_eps_ratio=0.01):
    if isinstance(pcd, o3d.geometry.PointCloud):
        points = np.asarray(pcd.points)
    elif isinstance(pcd, np.ndarray):
        print("Find contours Error: Input is not a PointCloud object.")
        return

    # 1. PCA 主方向（dir1, dir2 构建局部平面）
    pca = PCA(n_components=3)
    pca.fit(points)
    dir1, dir2 = pca.components_[0], pca.components_[1]
    center = pca.mean_

    # 2. 投影到主平面 (2D)
    projected_2d = np.dot(points - center, np.vstack([dir1, dir2]).T)

    # 3. 映射到像素图像
    min_xy = projected_2d.min(axis=0)
    norm_proj = ((projected_2d - min_xy) * scale).astype(np.int32)
    canvas_size = norm_proj.max(axis=0) + 10
    canvas = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)
    canvas[norm_proj[:, 1], norm_proj[:, 0]] = 255

    # 4. 提取并简化轮廓
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("没有找到轮廓")
        return

    contour = contours[0]
    epsilon = approx_eps_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True).squeeze()
    float2d = approx.astype(np.float32) / scale + min_xy

    # 5. 构建 3D 线段和法线向量
    line_segments_2d = []
    line_normals_2d = []

    line_segments_3d = []
    line_indices = []
    line_colors = []
    arrow_meshes = []

    for i in range(len(float2d)):
        pt1_2d = float2d[i]
        pt2_2d = float2d[(i + 1) % len(float2d)]  # 闭合

        # 线段方向和法线
        vec = pt2_2d - pt1_2d
        length = np.linalg.norm(vec)
        if length == 0:
            continue
        direction = vec / length
        normal_2d = np.array([-direction[1], direction[0]])

        line_segments_2d.append([pt1_2d, pt2_2d])
        line_normals_2d.append(normal_2d)

        # 中点和法线终点（2D）
        mid_2d = (pt1_2d + pt2_2d) / 2
        normal_end_2d = mid_2d + normal_2d * 0.02  # 可调长度

        # 投影回 3D 空间
        pt1_3d = center + pt1_2d[0]*dir1 + pt1_2d[1]*dir2
        pt2_3d = center + pt2_2d[0]*dir1 + pt2_2d[1]*dir2
        mid_3d = center + mid_2d[0]*dir1 + mid_2d[1]*dir2
        normal_end_3d = center + normal_end_2d[0]*dir1 + normal_end_2d[1]*dir2

        # 线段添加到 LineSet
        idx = len(line_segments_3d)
        line_segments_3d.extend([pt1_3d, pt2_3d])
        line_indices.append([idx, idx + 1])
        color = plt.cm.hsv(i / len(float2d))[:3]
        line_colors.append(color)

        # 法线箭头（arrow）
        arrow = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([mid_3d, normal_end_3d]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        arrow.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # 法线为绿色
        arrow_meshes.append(arrow)

    # 构建所有线段
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_segments_3d)
    line_set.lines = o3d.utility.Vector2iVector(line_indices)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    # 显示点云 + 所有线段 + 法线箭头
    o3d.visualization.draw_geometries([pcd, line_set] + arrow_meshes,window_name="P2 Contour Lines + Normals",width=1280, height=800)

    return line_segments_2d,line_normals_2d,dir1,dir2,center

contour_segments_2d_p2 = []
contour_normals_2d_p2 = []
contour_segments_2d_p2,contour_normals_2d_p2,dir1,dir2,center = extract_and_visualize_contour_segments_with_normals(proj_pcd_p2, scale=1500, approx_eps_ratio=0.01)


# **************************** Find and Show Initial TCP Box & Test Grid Point ****************************
def generate_grid_by_spacing(segments_2d, normals_2d, width=0.05, spacing=0.005):
    """
    每条线段沿法线方向扩展构造矩形，并在其中以 spacing 为间距生成等距网格点。
    
    参数：
        segments_2d: List of (pt1, pt2)，二维线段起止点
        normals_2d: List of unit normal vectors，每条线段一个
        width: 抓取区域宽度（法线方向），单位 m
        spacing: 网格点间隔（单位 m）
        
    返回：
        rectangles: 每个线段对应的矩形（4个点）
        all_grid_points: 每个矩形中生成的点，List[np.ndarray]
    """
    rectangles = []
    all_grid_points = []

    for (pt1, pt2), n in zip(segments_2d, normals_2d):
        pt1 = np.array(pt1)
        pt2 = np.array(pt2)
        n = np.array(n) / np.linalg.norm(n)

        # 线段方向和长度
        dir_vec = pt2 - pt1
        seg_len = np.linalg.norm(dir_vec)
        dir_unit = dir_vec / seg_len

        # 决定方向上的步数
        num_x = int(np.floor(seg_len / spacing))
        num_y = int(np.floor(width / spacing))
        if num_x < 1 or num_y < 1:
            continue

        # 构造矩形四个点（逆时针）
        offset = -n * width
        p1 = pt1 + offset
        p2 = pt2 + offset
        p3 = pt2
        p4 = pt1
        rectangles.append([p1, p2, p3, p4])

        # 在矩形内部生成规则点
        grid_pts = []
        for i in range(num_x):
            for j in range(num_y):
                alpha = (i + 0.5) * spacing
                beta = (j + 0.5) * spacing
                pt = p1 + dir_unit * alpha + n * beta
                grid_pts.append(pt)
        all_grid_points.append(np.array(grid_pts))

    return rectangles, all_grid_points

def plot_segments_tcpbox_and_grids(segments_2d, rectangles, grid_points):
    """
    在2D坐标平面绘制：
    - 原始线段（蓝色）
    - 每个线段的矩形区域（绿色虚线）
    - 矩形内部的规则网格点（红色 x）
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    used_labels = set()  # 追踪已添加的图例标签

    for (pt1, pt2), rect, grids in zip(segments_2d, rectangles, grid_points):
        # 原始线段
        lbl = 'Edges of Plane2'
        if lbl not in used_labels:
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=2, label=lbl)
            used_labels.add(lbl)
        else:
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=2)

        # 矩形区域（闭合）
        rect = np.array(rect + [rect[0]])
        lbl = 'Initial TCP Box'
        if lbl not in used_labels:
            ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=1, label=lbl)
            used_labels.add(lbl)
        else:
            ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=1)

        # 网格点
        grids = np.array(grids)
        lbl = 'Test Grid Points'
        if lbl not in used_labels:
            ax.plot(grids[:, 0], grids[:, 1], 'rx', markersize=4, label=lbl)
            used_labels.add(lbl)
        else:
            ax.plot(grids[:, 0], grids[:, 1], 'rx', markersize=4)

    ax.set_aspect('equal')
    ax.set_title("2D: Edges, TCP Box, and Test Grid Points")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


tcp_box,test_grid_points = generate_grid_by_spacing(contour_segments_2d_p2, contour_normals_2d_p2, width=c_pg, spacing=0.005)
plot_segments_tcpbox_and_grids(contour_segments_2d_p2,tcp_box,test_grid_points)

# Show each TCP Boxes and it's test grid points
def highlight_segment_rect_grid(segments_2d, rectangles, grid_points):
    """
    始终显示所有线段，只高亮当前索引对应的矩形与网格点。
    """
    all_pts = np.array(list(itertools.chain.from_iterable(segments_2d)))
    min_xy = all_pts.min(axis=0) - 0.02
    max_xy = all_pts.max(axis=0) + 0.02

    used_labels = set()

    for i in range(len(segments_2d)):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"Highlight Segment {i+1}/{len(segments_2d)}")

        # 所有线段：蓝色
        lbl = 'Edges of Plane2'
        for pt1, pt2 in segments_2d:
            if lbl not in used_labels:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=1,label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=1)

        # 当前的矩形：绿色
        rect = np.array(rectangles[i] + [rectangles[i][0]])
        ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=2, label='Initial TCP Box')

        # 当前的 grid 点：红色
        grids = np.array(grid_points[i])
        ax.plot(grids[:, 0], grids[:, 1], 'rx', markersize=4, label='Test Grid Points')

        ax.set_xlim(min_xy[0], max_xy[0])
        ax.set_ylim(min_xy[1], max_xy[1])
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


# highlight_segment_rect_grid(contour_segments_2d_p2, tcp_box, test_grid_points)


# Show Gripper Bounding Box
def create_gripper_bounding_box(grid_points, segments_2d):

    all_shapes = []
    segment_directions = [pt2 - pt1 for pt1, pt2 in segments_2d]

    for pts, seg_dir in zip(grid_points, segment_directions):
        seg_dir = seg_dir / np.linalg.norm(seg_dir)
        normal = np.array([-seg_dir[1], seg_dir[0]])
        segment_shapes = []

        for pt in pts:
            pt = np.array(pt)

            rectangles = []
            
            #Safespace Finger front
            center1 = pt - normal * x_pg
            p11 = center1 + seg_dir * (e_pg/2 + i_pg) 
            p12 = center1 + seg_dir * (e_pg/2 + i_pg) + normal * x_pg
            p13 = center1 - seg_dir * (e_pg/2 + i_pg) + normal * x_pg
            p14 = center1 - seg_dir * (e_pg/2 + i_pg)
            rectangles.append([p11, p12, p13, p14])

            #Finger length
            center2 = pt 
            p21 = center2 + seg_dir * (e_pg/2 + i_pg) 
            p22 = center2 + seg_dir * (e_pg/2 + i_pg) + normal * c_pg
            p23 = center2 - seg_dir * (e_pg/2 + i_pg) + normal * c_pg
            p24 = center2 - seg_dir * (e_pg/2 + i_pg)
            rectangles.append([p21, p22, p23, p24])

            #Gripper Base
            center3 = center2 + normal * c_pg
            p31 = center3 + seg_dir * (y_pg) 
            p32 = center3 + seg_dir * (y_pg) + normal * (j_pg - c_pg)
            p33 = center3 - seg_dir * (y_pg) + normal * (j_pg - c_pg)
            p34 = center3 - seg_dir * (y_pg)
            rectangles.append([p31, p32, p33, p34])

            #Robot Arm
            base_center = center3 + normal * (j_pg - c_pg)
            top_center = base_center + normal * lenth_robot

            b1 = base_center + seg_dir * radius_robot
            b2 = base_center - seg_dir * radius_robot
            t2 = top_center - seg_dir * (radius_robot + lenth_robot * math.tan(angle_robot))
            t1 = top_center + seg_dir * (radius_robot + lenth_robot * math.tan(angle_robot))

            trapezoid = [b1, b2, t2, t1]

            segment_shapes.append({
                'point': pt,
                'rectangles': rectangles,
                'trapezoid': trapezoid
            })

        all_shapes.append(segment_shapes)

    return all_shapes


def show_gripper_bounding_box(segments_2d, tcp_box, shapes):
    all_pts = [pt for seg in segments_2d for pt in seg]
    bounds = np.array(all_pts)
    min_xy = bounds.min(axis=0) - 0.01
    max_xy = bounds.max(axis=0) + 0.01

    for i, segment_shape in enumerate(shapes):
        for j, shape in enumerate(segment_shape):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(f"Segment {i+1}, Point {j+1}")

            used_labels = set()

            # 所有线段（蓝）
            lbl = 'Edges of Plane2'
            for pt1, pt2 in segments_2d:
                if lbl not in used_labels:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=1, label=lbl)
                    used_labels.add(lbl)
                else:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=1)

            # 初始矩形
            rect = np.array(tcp_box[i] + [tcp_box[i][0]])
            lbl = 'Initial TCP Box'
            if lbl not in used_labels:
                ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=2, label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=2)

            # 当前测试点
            pt = shape['point']
            lbl = 'Test Point'
            if lbl not in used_labels:
                ax.plot(pt[0], pt[1], 'ro', markersize=4, label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot(pt[0], pt[1], 'ro', markersize=4)

            # 三个矩形
            colors = ['red', 'purple', 'orange']
            Box_label = ['Finger Safe Space Box', 'Finger Box', 'Finger Base Box']
            for k, rect in enumerate(shape['rectangles']):
                poly = np.array(rect + [rect[0]])
                lbl = Box_label[k]
                if lbl not in used_labels:
                    ax.plot(poly[:, 0], poly[:, 1], color=colors[k], linewidth=1.5, label=lbl)
                    used_labels.add(lbl)
                else:
                    ax.plot(poly[:, 0], poly[:, 1], color=colors[k], linewidth=1.5)

            # 梯形
            trap = np.array(shape['trapezoid'] + [shape['trapezoid'][0]])
            lbl = 'Robot Arm'
            if lbl not in used_labels:
                ax.plot(trap[:, 0], trap[:, 1], color='deepskyblue', linestyle='--', linewidth=1.2, label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot(trap[:, 0], trap[:, 1], color='deepskyblue', linestyle='--', linewidth=1.2)

            ax.set_xlim(min_xy[0], max_xy[0])
            ax.set_ylim(min_xy[1], max_xy[1])
            ax.set_aspect('equal')
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.show()

show_gripper_bounding_box(contour_segments_2d_p2,tcp_box,create_gripper_bounding_box(test_grid_points,contour_segments_2d_p2))

#--------------------------find TCP area outside the collision area----------

if proj_pcd_outside.has_points() == False:
    print("No points in outside collision area!")
    candidate_TCP_remove_collsion_pcd = candidate_TCP_pcd
else:
    candidate_TCP_point = np.asarray(candidate_TCP_pcd.points)


    # # Methode 1 - cv2 range
    # projected_2d_out = np.dot(projected_points_outside, np.vstack([dir1, dir2]).T)
    # scale1 = 1000
    # min_xy1 = projected_2d_out.min(axis=0)
    # norm_points1 = ((projected_2d_out - min_xy1) * scale1).astype(np.int32)

    # canvas_size1 = norm_points1.max(axis=0) + 10
    # canvas1 = np.zeros((canvas_size1[1], canvas_size1[0]), dtype=np.uint8)
    # canvas1[norm_points1[:,1], norm_points1[:,0]] = 255

    # contours1, _ = cv2.findContours(canvas1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # expanded_polygons = []
    # x_expand_mm = 0.005  # e/2+i 外扩距离（以投影坐标单位为准）
    # for cnt in contours1:
    #     cnt = cnt.squeeze()
    #     poly = Polygon(cnt)
    #     expanded = poly.buffer(x_expand_mm * scale1)  # x_expand_mm 是外扩距离（以投影坐标单位为准）
    #     expanded_polygons.append(expanded)

    # projected_2d_2 = np.dot(candidate_TCP_point, np.vstack([dir1, dir2]).T)
    # projected_2d_2_norm = (projected_2d_2 - min_xy1)  # 无需缩放，保持原始比例

    # mask = []
    # for pt in projected_2d_2_norm:
    #     p = Point(pt)
    #     in_any = any(expanded.contains(p) for expanded in expanded_polygons)
    #     mask.append(not in_any)  # 不在任何外扩区才保留
    # mask = np.array(mask)

    # candidate_TCP_remove_collsion_point = candidate_TCP_point[mask]

    # candidate_TCP_remove_collsion_pcd = o3d.geometry.PointCloud()
    # candidate_TCP_remove_collsion_pcd.points = o3d.utility.Vector3dVector(candidate_TCP_remove_collsion_point)

    # # 点云1外扩轮廓可视化（还原到3D）
    # lines = []
    # for poly in expanded_polygons:
    #     if poly.is_empty:
    #         continue
    #     exterior = np.array(poly.exterior.coords)
    #     # 转回原始尺度
    #     exterior_2d = exterior
    #     # 投影回3D
    #     exterior_3d = [x * dir1 + y * dir2 + pca.mean_ for x, y in exterior_2d]
    #     # 连接成线段
    #     for i in range(len(exterior_3d) -1):
    #         lines.append((exterior_3d[i], exterior_3d[i+1]))

    # line_points = []
    # line_indices = []
    # idx = 0
    # for p1, p2 in lines:
    #     line_points.append(p1)
    #     line_points.append(p2)
    #     line_indices.append([idx, idx+1])
    #     idx +=2

    # line_set = o3d.geometry.LineSet()
    # line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
    # line_set.lines = o3d.utility.Vector2iVector(line_indices)
    # line_set.paint_uniform_color([0, 1, 1])  # 红色框





    # # Methode 2 - point to point comparison
    candidate_TCP_remove_collsion_pcd = extract_overlap_region(candidate_TCP_pcd, proj_pcd_outside, threshold=0.001,remove=True)

    candidate_TCP_remove_collsion_pcd.paint_uniform_color([0, 1, 0]) 
o3d.visualization.draw_geometries([candidate_TCP_remove_collsion_pcd,boundary_pcd.translate((0,0,0.0001)),proj_pcd_outside],window_name="TCP/Inside Collision/Outside Collision")

#---------------------------Remove outliers & voxel sample ----------------------------------


# # 先体素降采样
# voxel_pcd = boundary_pcd.voxel_down_sample(voxel_size=0.002)

# # 然后再用最远点采样增强均匀性
# sampled_pcd = voxel_pcd.farthest_point_down_sample(num_samples=len(voxel_pcd.points))

# o3d.visualization.draw_geometries([sampled_pcd])

def statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    统计离群点去除
    :param pcd: 输入点云
    :param nb_neighbors: 邻域点数
    :param std_ratio: 标准差乘数阈值
    :return: 滤波后的点云
    """
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return cl

filtered_pcd = statistical_outlier_removal(candidate_TCP_remove_collsion_pcd)
o3d.visualization.draw_geometries([filtered_pcd],window_name="Statistical Outlier Removal CP")

# #---------------------------average grid----------------------------------
def regular_grid_sample(pcd, spacing):
    points = np.asarray(pcd.points)
    
    # 找点云边界
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    
    # 生成网格点
    xs = np.arange(min_bound[0], max_bound[0], spacing)
    ys = np.arange(min_bound[1], max_bound[1], spacing)
    zs = np.arange(min_bound[2], max_bound[2], spacing)
    
    grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1,3)
    
    # 用KDTree找网格点是否在点云范围（最近邻距离）
    tree = cKDTree(points)
    dist, _ = tree.query(grid, distance_upper_bound=spacing/2)
    
    mask = dist < spacing/2
    sampled_points = grid[mask]
    
    # 生成点云
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    
    return sampled_pcd

# 用法
spacing = 0.003  # 1cm间距
sampled_pcd = regular_grid_sample(filtered_pcd, spacing)
sampled_pcd.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([sampled_pcd],window_name="Grid TCP")


#--------------------------GSS-------------------------


#-----------center score------
def colorize_by_distance(sampled_pcd, center_point):
    points = np.asarray(sampled_pcd.points)
    
    # 计算每个点到中心的距离
    distances = np.linalg.norm(points - center_point, axis=1)
    
    # 归一化到 [0,1]
    min_dist = distances.min()
    max_dist = distances.max()
    normalized = 1 - (distances - min_dist) / (max_dist - min_dist + 1e-8)  # 防止除零

    # 红(0) -> 绿(1) 线性插值
    colors = np.zeros((len(points), 3))
    colors[:,0] = 1 - normalized   # 红色分量
    colors[:,1] = normalized       # 绿色分量
    colors[:,2] = 0                # 无蓝色

    # 赋给点云
    sampled_pcd.colors = o3d.utility.Vector3dVector(colors)

    return sampled_pcd


#--------cavature score--------

# 用法
center_point =  np.mean(np.asarray(pcd.points), axis=0)
colored_pcd = colorize_by_distance(sampled_pcd, center_point)
pcd.paint_uniform_color([0.5, 0.5, 0.5])

sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
sphere.paint_uniform_color([0, 0, 1])  # 蓝色
sphere.translate(center_point)

o3d.visualization.draw_geometries([colored_pcd,pcd,sphere],window_name="GSS Ranking Result")
