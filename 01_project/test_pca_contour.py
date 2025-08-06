import open3d as o3d
import numpy as np
import random
import copy
from math import acos, degrees
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import math

import alphashape
from shapely.geometry import Point, MultiPoint,Polygon,MultiLineString, MultiPolygon
import matplotlib.pyplot as plt
from scipy.spatial import KDTree,cKDTree
import cv2
import itertools

from scipy.spatial import ConvexHull, Delaunay
from shapely.ops import unary_union, polygonize

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
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if approx is None or len(approx) < 2:
        print("轮廓点不足")
        return
    approx = approx.reshape(-1, 2)  # 明确为 (N, 2) 的形状
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

        print(type(direction))
        print(direction)


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

pcd = o3d.io.read_point_cloud(r"D:\Codecouldcode\099.MA_Hanyu\overlap_pcd.pcd")
extract_and_visualize_contour_segments_with_normals(pcd,250)