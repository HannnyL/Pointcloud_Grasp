import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from matplotlib.path import Path

object_pcd = o3d.io.read_point_cloud("D:\\Codecouldcode\\099.MA_Hanyu\\Object\\object_04.pcd")

plane_model = [-0.0000,1.0000,0.0000,-0.0250]
a, b, c, d = plane_model
sample_resolution = 50

def project_points_to_plane(pcd, plane_model):
    points = np.asarray(pcd.points)
    a, b, c, d = plane_model
    plane_normal = np.array([a, b, c])
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # 计算点到平面的距离（标量）
    dist_to_plane = (points @ plane_normal) + d  # shape (N,)
    
    # 沿法向方向减去距离，实现投影
    projected_points = points - np.outer(dist_to_plane, plane_normal)

    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(projected_points)
    return projected_pcd


# 投影物体点云到底面，得到下底面点云
projected_pcd = project_points_to_plane(object_pcd, plane_model)


# 合并原始点云和投影点云，实现“补全下底面”
combined_points = np.vstack([np.asarray(object_pcd.points), np.asarray(projected_pcd.points)])
combined_pcd = o3d.geometry.PointCloud()
combined_pcd.points = o3d.utility.Vector3dVector(combined_points)

# 之后对 combined_pcd 计算质心等信息
centroid = np.mean(np.asarray(combined_pcd.points), axis=0)
print("补全下底面后质心:", centroid)

# 可视化验证
o3d.visualization.draw_geometries([combined_pcd])
