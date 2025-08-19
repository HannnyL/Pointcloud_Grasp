import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path

# ---------- 参数设置 ----------
pcd_path = "D:\\Codecouldcode\\099.MA_Hanyu\\Object\\object_03.pcd"  # 单个物体点云路径
a, b, c, d = -0.0000,1.0000,0.0000,-0.0250  # 桌面平面方程 ax + by + cz + d = 0
delta = 0.0015  # 与桌面垂直距离阈值（5mm）
sample_resolution = 50  # 底面补全分辨率
normal_thresh = 0.9  # 法向朝下判断阈值（dot值）
# ------------------------------

# 桌面法向
plane_normal = np.array([a, b, c])
plane_normal = plane_normal / np.linalg.norm(plane_normal)

# 加载点云
pcd = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(pcd.points)

# 估算法向量
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.orient_normals_to_align_with_direction(orientation_reference=-plane_normal)  # 使法向朝下
normals = np.asarray(pcd.normals)

# 计算点到平面距离
dist_to_plane = points @ plane_normal + d

# 底部点筛选（距离近 + 法向朝下）
close_mask = np.abs(dist_to_plane) < delta
downward_mask = normals @ (-plane_normal) > normal_thresh
bottom_mask = close_mask & downward_mask
bottom_points = points[bottom_mask]

print(f"原始点数: {len(points)}, 底部点数: {len(bottom_points)}")

if len(bottom_points) < 20:
    print("底部点太少，跳过补全")
    exit()

# 底部点投影到平面
bottom_proj = bottom_points - np.outer(dist_to_plane[bottom_mask], plane_normal)
xz_points = bottom_proj[:, [0, 2]]  # 投影到XZ平面（Y为竖轴）

# 凸包轮廓
hull = ConvexHull(xz_points)
hull_vertices = xz_points[hull.vertices]

# 网格采样
xmin, zmin = hull_vertices.min(axis=0)
xmax, zmax = hull_vertices.max(axis=0)
xx, zz = np.meshgrid(
    np.linspace(xmin, xmax, sample_resolution),
    np.linspace(zmin, zmax, sample_resolution)
)
grid_xz = np.vstack([xx.ravel(), zz.ravel()]).T

# 判断是否在凸包内
path = Path(hull_vertices)
inside_mask = path.contains_points(grid_xz)
filled_xz = grid_xz[inside_mask]

# 恢复对应Y坐标（平面上）
x_fill = filled_xz[:, 0]
z_fill = filled_xz[:, 1]
y_fill = -(a * x_fill + c * z_fill + d) / b
filled_points = np.vstack([x_fill, y_fill, z_fill]).T

# 合并点云
final_points = np.vstack([points, filled_points])
pcd_out = o3d.geometry.PointCloud()
pcd_out.points = o3d.utility.Vector3dVector(final_points)

# 可视化
o3d.visualization.draw_geometries([pcd_out])

# 可选保存
# o3d.io.write_point_cloud("object_filled.pcd", pcd_out)

