import open3d as o3d
from sklearn.decomposition import PCA
import numpy as np
import alphashape
from shapely.geometry import Polygon

import matplotlib.pyplot as plt


pcd = o3d.io.read_point_cloud(r"D:\Codecouldcode\099.MA_Hanyu\proj_pcd_p4.pcd")
points = np.asarray(pcd.points)

# 1. PCA 主方向（dir1, dir2 构建局部平面）
pca = PCA(n_components=3)
pca.fit(points)
dir1, dir2 = pca.components_[0], pca.components_[1]
center = pca.mean_

# 2. 投影到主平面 (2D)
projected_2d = np.dot(points - center, np.vstack([dir1, dir2]).T)


# alpha = alphashape.optimizealpha(projected_2d)
alpha = 0.01
hull = alphashape.alphashape(projected_2d, alpha)

# 提取边界坐标
exterior_coords = np.array(hull.exterior.coords)




plt.scatter(projected_2d[:,0], projected_2d[:,1], s=1, color='blue')
plt.plot(exterior_coords[:,0], exterior_coords[:,1], color='red')
plt.axis('equal')
plt.show()

