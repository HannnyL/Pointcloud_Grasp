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
from shapely.geometry import Point, MultiPoint,Polygon,MultiLineString, MultiPolygon,LineString
import matplotlib.pyplot as plt
from scipy.spatial import KDTree,cKDTree
import cv2
import itertools

from scipy.spatial import ConvexHull, Delaunay
from shapely.ops import unary_union, polygonize


def remove_shell(pcd,scale=1000,thickness=0.001):

    points = np.asarray(pcd.points)

    
    pca = PCA(n_components=3)
    pca.fit(points)
    dir1, dir2 = pca.components_[0], pca.components_[1]
    center = pca.mean_

    # 将点投影到 PCA 提取的主方向1 和 主方向2 上，得到2D坐标
    projected_2d = np.dot(points - center, np.vstack([dir1, dir2]).T)


    # 2. 归一化到像素坐标

    # 扩大比例，保证像素精度
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
    # 1cm 宽度
    inner_poly = polygon.buffer(-thickness * scale)
    outer_poly = polygon

    # 6. 遍历所有投影点，筛选在边界带的
    mask = []
    for pt in norm_points:
        p = Point(pt)
        if inner_poly.contains(p) and outer_poly.contains(p):
            mask.append(True)
        else:
            mask.append(False)
    mask = np.array(mask)

    remove_shell_points = points[mask]

    # 7. 显示
    # boundary_pcd = o3d.geometry.PointCloud()
    # boundary_pcd.points = o3d.utility.Vector3dVector(remove_shell_points)
    # boundary_pcd.paint_uniform_color([0,1,0])

    pcd.paint_uniform_color([1,0,0])
    colors = np.asarray(pcd.colors)
    colors[mask] = [0,1,0]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd],window_name="remove shell")
    print("scale:",scale)

if __name__ == '__main__':

    scale = 1000
    thickness = 0.001

    pcd = o3d.io.read_point_cloud(r"D:\Codecouldcode\099.MA_Hanyu\proj_pcd_p4.pcd")

    count = 1
    while(True):
        print("count:",count)
        try:
            
            remove_shell(pcd,scale,thickness)
            break
        except:
            scale -= 800
            count += 1

            if count > 10:
                break