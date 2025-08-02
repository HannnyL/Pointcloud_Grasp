import open3d as o3d
import numpy as np
import random
import copy
from math import acos, degrees
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from collections import Counter

import alphashape
from shapely.geometry import Point, MultiPoint,Polygon
import matplotlib.pyplot as plt
from scipy.spatial import KDTree,cKDTree
import cv2
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

        b_pg = 0.01 # TCP to Finger length end
        c_pg = 0.03 # TCP to Safety space of Gripper
        d_pg = space # Safespace Gripper length
        x_pg = space # Safespace Gripper end to rubber
        n_pg = d_pg + c_pg + b_pg # Finger length
        t_pg = 0.065 # Gripper base bottom length
        u_pg = 0.05 # Gripper base top length
        j_pg = c_pg + d_pg + t_pg + u_pg # Gripper length (TCP to Robot)
        s_pg = j_pg + b_pg + x_pg # Total gripper length

        e_pg = 0.04 # Finger depth
        i_pg = space # Safespace finger depth
        l_pg = 0.06 # Gripper base bottom depth
        m_pg = space # Safespace gripper base bottom depth
        o_pg = 0.07 # Gripper base top  depth
        p_pg = space # Safespace gripper base top depth


space = 0.005
a_pg = 0.01 # Finger width
w_pg = 0.3*space # Internal Safespace Finger width 
v_pg = space # External Safespace Finger width 
f_pg = 0.10 # Distance gripper open
g_pg = 0.02 # Distance gripper close
h_pg = 0.12 # Gripper base width
k_pg = space # Safespace Gripper base width 
q_pg = 0.08 # Robot end width
r_pg = space # Safespace Robot width

b_pg = 0.01 # TCP to Finger length end
c_pg = 0.03 # TCP to Safety space of Gripper
d_pg = space # Safespace Gripper length
n_pg = d_pg + c_pg + b_pg # Finger length
t_pg = 0.065 # Gripper base length
u_pg = 0.05 # Robot end length
j_pg = c_pg + d_pg + t_pg # Gripper length (TCP to Robot)
s_pg = j_pg + b_pg # Total gripper length

e_pg = 0.04 # Finger depth
i_pg = space # Safespace finger depth
l_pg = 0.06 # Gripper base depth
m_pg = space # Safespace gripper base depth
o_pg = 0.07 # Robot end depth
p_pg = space # Safespace robot end depth



# ---------------- Step 1: 读取点云+外法线估计 ----------------
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

#------------------------Project planes and find overlap region----------

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

o3d.visualization.draw_geometries([
    overlap_pcd,pcd_orig_i,pcd_orig_j

],window_name="Pair of Planes and Their Overlap Region")

#--------------------------------Find points between planes------------
def project_points_to_plane(points, plane_point, plane_normal):
    v = points - plane_point
    d = np.dot(v, plane_normal)
    return points - np.outer(d, plane_normal)

def select_points_between_planes(pcd, center_i, center_j, plane_normal, margin=0.0005, include_planes=True):
    """
    筛选出完整点云中夹在两个平面之间的点
    """
    points = np.asarray(pcd.points)
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
    return points_between




# 1. 筛选出在两个平面之间的点
points_between = select_points_between_planes(pcd, center_i, center_j, plane_normals[mmm])

# 2. 投影到中间平面
projected_points = project_points_to_plane(points_between, center_ij, plane_normals[mmm])

# 3. 创建 PointCloud 对象
proj_pcd = o3d.geometry.PointCloud()
proj_pcd.points = o3d.utility.Vector3dVector(projected_points)
proj_pcd.paint_uniform_color([1, 0, 0])  # 红色


# 可视化所有内容
o3d.visualization.draw_geometries([
    pcd_orig_j, pcd_orig_i, proj_pcd
    ,
],window_name="Pair of Planes and Projected Points Between Them")

#---------------------------------Alpha shape----------

# 加载点云
# projected_points  平面中间的点投影至中间平面的点 nparray
# proj_pcd 转换为点云对象

# voxel_size = 0.005  # 降采样率
# proj_down = proj_pcd.voxel_down_sample(voxel_size)
# down_points = np.asarray(proj_down.points)
# o3d.visualization.draw_geometries([proj_down])


# 1. PCA获取两个主要方向

pca = PCA(n_components=3)
pca.fit(projected_points)

# 主轴（单位向量）
dir1 = pca.components_[0]
dir2 = pca.components_[1]
center = pca.mean_

#-----------------------chat-cv2------------------------

# 将点投影到 PCA 提取的主方向1 和 主方向2 上，得到2D坐标
projected_2d = np.dot(projected_points - center, np.vstack([dir1, dir2]).T)

# 2. 归一化到像素坐标
scale = 1000  # 扩大比例，保证像素精度
min_xy = projected_2d.min(axis=0)
norm_points = ((projected_2d - min_xy) * scale).astype(np.int32)

canvas_size = norm_points.max(axis=0) + 10
canvas = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)
canvas[norm_points[:,1], norm_points[:,0]] = 255

# 3. 提取轮廓
contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. 轮廓转 shapely polygon
cnt = contours[0].squeeze()
polygon = Polygon(cnt)

# 5. buffer 内缩 & 外扩，形成宽度
thickness = 0.01  # 1cm 宽度
inner_poly = polygon.buffer(-thickness * scale)
outer_poly = polygon

# 6. 遍历所有投影点，筛选在边界带的
mask = []
for pt in norm_points:
    p = Point(pt)
    if outer_poly.contains(p) and not inner_poly.contains(p):
        mask.append(True)
    else:
        mask.append(False)
mask = np.array(mask)

boundary_points = projected_points[mask]

# 7. 显示
boundary_pcd = o3d.geometry.PointCloud()
boundary_pcd.points = o3d.utility.Vector3dVector(boundary_points)
boundary_pcd.paint_uniform_color([1,0,0])

o3d.visualization.draw_geometries([boundary_pcd],window_name="Cadidate TCP with Inside Collision Boundary")


# #---------------------transform-------------------
# trans_dist = 0.015  # 平移距离
# # 2. 沿两个主方向正向平移3mm
# shift_vector = trans_dist * (dir1 + dir2)  # 3cm 移动向量
# shifted_points = projected_points + shift_vector

# # 3. 与原点云求交集（半径2mm）
# radius_tree = 0.0005
# tree = KDTree(projected_points)
# indices = tree.query_ball_point(shifted_points, radius_tree)
# intersection_mask = np.array([len(pts) > 0 for pts in indices])
# intersection_points = shifted_points[intersection_mask]

# intersection_pcd = o3d.geometry.PointCloud()
# intersection_pcd.points = o3d.utility.Vector3dVector(intersection_points)
# intersection_pcd.paint_uniform_color([1, 0, 0])  # 红色

# # 4. 再向两个主方向负方向平移3mm
# second_shift_vector = -trans_dist * (dir1 + dir2)
# second_shifted_points = intersection_points + second_shift_vector

# # 与上次交集点云求交集
# tree2 = KDTree(intersection_points)
# indices2 = tree2.query_ball_point(second_shifted_points, radius_tree)
# second_mask = np.array([len(pts) > 0 for pts in indices2])
# second_intersection_points = second_shifted_points[second_mask]

# second_pcd = o3d.geometry.PointCloud()
# second_pcd.points = o3d.utility.Vector3dVector(second_intersection_points)
# second_pcd.paint_uniform_color([0, 1, 0])  # 绿色

# # 5. 原始点云去除二次交集，得到边界框
# tree_orig = KDTree(projected_points)
# mask_boundary = np.ones(len(projected_points), dtype=bool)
# remove_indices = tree_orig.query_ball_point(second_intersection_points, radius_tree)
# for idx_list in remove_indices:
#     for idx in idx_list:
#         mask_boundary[idx] = False

# boundary_points = projected_points[mask_boundary]
# boundary_pcd = o3d.geometry.PointCloud()
# boundary_pcd.points = o3d.utility.Vector3dVector(boundary_points)
# boundary_pcd.paint_uniform_color([0, 0, 1])  # 蓝色

# # 6. 显示
# # o3d.visualization.draw_geometries([intersection_pcd,second_pcd,boundary_pcd])
# o3d.visualization.draw_geometries([boundary_pcd])


# --------------------alpha shape 方法
# '''
# # 将点投影到 PCA 提取的主方向1 和 主方向2 上，得到2D坐标
# projected_2d = np.dot(down_points, np.vstack([dir1, dir2]).T)

# # 4. 用 alphashape 计算不规则边界
# alpha = 0.0001  # alpha值需视点云密度调整
# alpha_shape = alphashape.alphashape(projected_2d, alpha)

# # 2. 检查并修复闭合性
# if not alpha_shape.is_valid or alpha_shape.is_empty or not alpha_shape.is_closed:
#     print("Alpha shape 有缺口或无效，正在尝试修复...")
#     alpha_shape = alpha_shape.buffer(0)

# if alpha_shape.is_valid and alpha_shape.is_closed:
#     print("轮廓已闭合修复")
# else:
#     print("修复失败，轮廓仍未闭合")

# # 3. 可视化检查（Matplotlib 显示2D轮廓）
# x, y = zip(*projected_2d)
# plt.scatter(x, y, s=1, color='blue', label='Points')

# if hasattr(alpha_shape, 'exterior'):
#     boundary_x, boundary_y = alpha_shape.exterior.xy
#     plt.plot(boundary_x, boundary_y, color='red', label='Alpha Shape Boundary')

# plt.legend()
# plt.show()

# # 5. buffer 缩小1cm
# buffer_distance = -0.01
# shrunk_shape = alpha_shape.buffer(buffer_distance)

# # 6. 筛选边界带的点：在原边界内但不在收缩后边界内
# boundary_mask = []
# for pt in projected_2d:
#     p = Point(pt)
#     in_outer = alpha_shape.contains(p)
#     in_inner = shrunk_shape.contains(p)
#     boundary_mask.append(in_outer and not in_inner)
# boundary_mask = np.array(boundary_mask)

# boundary_points = down_points[boundary_mask]

# # 7. 生成边界点云并上色
# boundary_pcd = o3d.geometry.PointCloud()
# boundary_pcd.points = o3d.utility.Vector3dVector(boundary_points)
# boundary_pcd.paint_uniform_color([0.5, 0.5, 0.5]) 

# # 8. 可视化

# o3d.visualization.draw_geometries([boundary_pcd])
# '''

#-------------------find concave corners----------------

# 假设 contours[0] 是外轮廓（来自你的代码）
contours_2, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_2 = contours_2[0]  # 外轮廓点（形状为 (N,1,2)）

# 1. 简化轮廓（可选，减少噪声点）
epsilon = 0.01 * cv2.arcLength(contour_2, True)
approx_contour_2 = cv2.approxPolyDP(contour_2, epsilon, True)

# 2. 检测凹点
def find_concave_points(contour):
    contour = contour.squeeze()  # 去除冗余维度 (N,2)
    n = len(contour)
    concave_points = []
    convex_points = []

    print(f"contours 有{n}个！！")
    
    for i in range(n):
        p_prev = contour[(i-1) % n]
        p_curr = contour[i]
        p_next = contour[(i+1) % n]
        
        # 计算向量
        u = p_curr - p_prev
        v = p_next - p_curr
        
        # 叉积的z分量（判断凹凸性）
        cross = u[0] * v[1] - u[1] * v[0]
        
        if cross > 0:  # 凹点（假设轮廓点顺时针排列）
            concave_points.append(p_curr)
        elif cross < 0:  # 凸点
            convex_points.append(p_curr)
    
    return np.array(concave_points), np.array(convex_points)

# 调用函数
concave_points_pixel,convex_points_pixel = find_concave_points(approx_contour_2)

if concave_points_pixel.size == 0:
    print("No concave points found!")
    o3d.visualization.draw_geometries([boundary_pcd],window_name="Candidate TCP with no concave points found!")
else:
    # 1. 确保 concave_points_pixel 是 Nx2 的数组
    concave_points_pixel = np.array(concave_points_pixel)
    if concave_points_pixel.ndim == 3:  # 如果是 (N,1,2)，则压缩为 (N,2)
        concave_points_pixel = concave_points_pixel.squeeze(axis=1)

    convex_points_pixel = np.array(convex_points_pixel)
    if convex_points_pixel.ndim == 3:  # 如果是 (N,1,2)，则压缩为 (N,2)
        convex_points_pixel = convex_points_pixel.squeeze(axis=1)

    # 2. 像素坐标 -> 归一化2D
    concave_points_norm = concave_points_pixel / scale + min_xy  # (N,2)

    convex_points_norm = convex_points_pixel / scale + min_xy  # (N,2)

    # 3. 检查PCA主方向
    print("dir1 shape:", dir1.shape)  # 应该是(3,)
    print("dir2 shape:", dir2.shape)  # 应该是(3,)

    # 4. 构建A矩阵
    A = np.column_stack([dir1, dir2])  # 3x2矩阵+
    print("A shape:", A.shape)  # 应该是(3,2)

    # 5. 准备b矩阵
    b = concave_points_norm.T  # 2xN矩阵
    print("b shape:", b.shape)  # 应该是(2,N)

    b = convex_points_norm.T  # 2xN矩阵
    print("b shape:", b.shape)  # 应该是(2,N)

    # 6. 检查是否有足够的数据点
    if concave_points_norm.shape[0] < 1:
        raise ValueError("No concave points found!")

    # 1. 确保数据形状正确
    print("调试信息：")
    print("A shape:", A.shape)  # 应该是 (3,2)
    print("A_pinv shape:", np.linalg.pinv(A).shape)  # 应该是 (2,3)
    print("concave_points_norm shape:", concave_points_norm.shape)  # 应该是 (N,2)

    # 2. 正确的反投影计算
    # 方法一：使用最小二乘法（推荐）
    concave_points_3d = []
    convex_points_3d = []
    # for pt in concave_points_norm:
    #     # 每个点单独计算
    #     x = np.linalg.lstsq(A, pt, rcond=None)[0]
    #     concave_points_3d.append(x)
    # concave_points_3d = np.array(concave_points_3d)

    # 方法二：正确的矩阵运算方式（需确保形状匹配）
    A_pinv = np.linalg.pinv(A)  # (2,3)
    concave_points_3d = (concave_points_norm @ A_pinv) + center  # (N,2) @ (2,3) = (N,3)

    convex_points_3d = (convex_points_norm @ A_pinv) + center  # (N,2) @ (2,3) = (N,3)

    # 3. 验证结果
    print("反投影结果形状:", concave_points_3d.shape)  # 应该是 (N,3)
    assert concave_points_3d.shape[1] == 3, "反投影结果维度错误"

    # # 4. 可视化
    # concave_pcd = o3d.geometry.PointCloud()
    # concave_pcd.points = o3d.utility.Vector3dVector(concave_points_3d)
    # concave_pcd.paint_uniform_color([0, 1, 0])  # 绿色

    def create_spheres_at_points(points, radius=0.05, color=[1, 0, 0]):
        """
        在指定点位置创建球体标记
        参数：
            points: (N,3) numpy数组，球心位置
            radius: 球体半径
            color: [R,G,B]颜色，取值范围0-1
        返回：
            Open3D球体对象的列表
        """
        spheres = []
        for point in points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.paint_uniform_color(color)
            sphere.translate(point)
            spheres.append(sphere)
        return spheres

    concave_spheres = create_spheres_at_points(concave_points_3d,radius=0.002,color=[0, 1, 0])  # 绿色)
    convex_spheres = create_spheres_at_points(convex_points_3d,radius=0.002,color=[0, 0, 1])  # 蓝色)

    # o3d.visualization.draw_geometries([boundary_pcd, convex_spheres])

    all_geometries = [boundary_pcd]  # 点云对象
    all_geometries.extend(concave_spheres)  # 添加所有球体网格
    all_geometries.extend(convex_spheres)  # 添加所有球体网格

    o3d.visualization.draw_geometries(all_geometries,window_name="All Concave Points on TCP")

#----------------------overlap region collisionfree & projected plane----------
candidate_TCP_pcd = extract_overlap_region(overlap_pcd, boundary_pcd, threshold=0.001)
candidate_TCP_pcd.paint_uniform_color([0, 1, 0]) 
o3d.visualization.draw_geometries([candidate_TCP_pcd,boundary_pcd.translate((0,0,0.0001))],window_name="Candidate TCP overlap with Inside Collision Boudary")

#----------------------find outside collision area---------------------------------

center_i_outside = center_i + (a_pg + w_pg + v_pg) * (plane_normals[mmm])
center_j_outside = center_j + (a_pg + w_pg + v_pg) * (plane_normals[nnn])

# 1. 筛选出在两个平面之间的点
points_between_outside_i = select_points_between_planes(pcd, center_i, center_i_outside, plane_normals[mmm],0.001,False)
points_between_outside_j = select_points_between_planes(pcd, center_j, center_j_outside, plane_normals[nnn],0.001,False)
points_between_outside = np.vstack((points_between_outside_i, points_between_outside_j))

# 2. 投影到中间平面
projected_points_outside = project_points_to_plane(points_between_outside, center_ij, plane_normals[mmm])

# 3. 创建 PointCloud 对象
proj_pcd_outside = o3d.geometry.PointCloud()
proj_pcd_outside.points = o3d.utility.Vector3dVector(projected_points_outside)
proj_pcd_outside.paint_uniform_color([0, 0, 1])  # 蓝色

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
o3d.visualization.draw_geometries([candidate_TCP_pcd, proj_pcd_outside],window_name="Candidate TCP & Outside Collision Area")

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
