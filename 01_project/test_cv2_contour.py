import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from shapely.geometry import Point, Polygon

# 生成矩形区域内均匀分布的点云
def generate_rectangle_points(center, width, height, num_points):
    x_points = np.random.uniform(center[0] - width/2, center[0] + width/2, num_points)
    y_points = np.random.uniform(center[1] - height/2, center[1] + height/2, num_points)
    points = np.vstack((x_points, y_points)).T
    return points

# 判断点是否在多边形内
def is_point_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, measureDist=False) >= 0

# 计算两个多边形的相交面积
def intersection_area(poly1, poly2):
    poly1 = poly1.astype(np.int32)
    poly2 = poly2.astype(np.int32)
    img_shape = (1000, 1000)  # 假设图像足够大
    mask1 = np.zeros(img_shape, dtype=np.uint8)
    mask2 = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask1, [poly1], 1)
    cv2.fillPoly(mask2, [poly2], 1)
    intersection = cv2.bitwise_and(mask1, mask2)
    return np.sum(intersection)

# 两个矩形的参数
rect1_center = [0, 0]
rect2_center = [3, 2]
rect_width, rect_height = 2, 1
num_points_rect1 = 500
num_points_rect2 = 500

# 生成两个矩形区域的点
rect1_points = generate_rectangle_points(rect1_center, rect_width, rect_height, num_points_rect1)
rect2_points = generate_rectangle_points(rect2_center, rect_width, rect_height, num_points_rect2)

# 合并两个矩形区域的点
# points = np.vstack((rect1_points, rect2_points))
pcd = o3d.io.read_point_cloud(r"D:\Codecouldcode\099.MA_Hanyu\proj_pcd_p4.pcd")
points = np.asarray(pcd.points)

# 1. PCA 主方向（dir1, dir2 构建局部平面）
pca = PCA(n_components=3)
pca.fit(points)
dir1, dir2 = pca.components_[0], pca.components_[1]
center = pca.mean_

# 2. 投影到主平面 (2D)
points = np.dot(points - center, np.vstack([dir1, dir2]).T)


# 转换点为图像坐标
img_scale = 1500
points_img = np.int32((points - points.min(axis=0)) * img_scale)
img_size = points_img.max(axis=0) + 10

# 创建空白图像并绘制点
img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
for pt in points_img:
    cv2.circle(img, tuple(pt), 1, 255, -1)

# 使用findContours寻找轮廓
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制各种轮廓处理结果
img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    # # 1. 轴对齐外接矩形（绿色）
    # x, y, w, h = cv2.boundingRect(cnt)
    # cv2.rectangle(img_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # # 2. 最小外接旋转矩形（蓝色）
    # rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    # box = box.astype(np.int32)
    # cv2.drawContours(img_contours, [box], 0, (255, 0, 0), 2)

    # # 3. 凸包轮廓（红色）
    # hull = cv2.convexHull(cnt)
    # cv2.drawContours(img_contours, [hull], 0, (0, 0, 255), 2)

    # 4. 多边形近似（黄色）
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(img_contours, [approx], 0, (0, 255, 255), 2)

    # # 5. 示例：判断点是否在轮廓内
    # test_point = (30, 20)  # 图像坐标
    # inside = is_point_inside_polygon(test_point, approx)
    # color = (0, 255, 0) if inside else (0, 0, 255)
    # cv2.circle(img_contours, test_point, 5, color, -1)

    # # 6. 示例：矩形区域与轮廓相交面积
    # test_rect = np.array([[17,18], [50,18], [50,25], [17,25]])
    # area = intersection_area(test_rect, approx)
    # cv2.polylines(img_contours, [test_rect], isClosed=True, color=(255, 255, 0), thickness=2)
    # print(f"Intersection area with test rectangle: {area}")


# 显示轮廓和高级处理结果
plt.figure(figsize=(10, 6))
plt.imshow(img_contours)
plt.title('Contours: Rect (Green), RotRect (Blue), Hull (Red), ApproxPoly (Yellow)')
plt.axis('off')
plt.show()




# ---- 将轮廓转换回原始坐标空间 ---- #
contour_points_list = []
linesets = []

for cnt in contours:
    # 近似多边形轮廓
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)

    # 从图像坐标系转换回投影的二维坐标系
    points_2d_back = approx.astype(np.float32) / img_scale + points.min(axis=0)

    # 从二维坐标映射回原始三维空间
    points_3d = np.dot(points_2d_back, np.vstack([dir1, dir2])) + center

    contour_points_list.append(points_3d)

    # 构造Open3D LineSet对象
    num_points = points_3d.shape[0]
    lines = [[i, (i+1)%num_points] for i in range(num_points)]

    colors = [[1, 0, 0] for _ in lines]  # 红色线条

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_3d)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    linesets.append(line_set)

# 原始点云设置颜色便于观察
pcd.paint_uniform_color([0.5, 0.5, 0.5])

# 使用Open3D可视化所有轮廓
o3d.visualization.draw_geometries([pcd, *linesets])





# 轮廓转回真实二维坐标
contour_real = approx.reshape(-1, 2).astype(np.float32)/img_scale + points.min(axis=0)
poly_contour = Polygon(contour_real)

# 判断点是否在轮廓内
test_point = (0.0, 0.02)
is_inside = poly_contour.contains(Point(test_point))
print(f"点{test_point}是否位于轮廓内：{is_inside}")

# 计算相交面积（实际单位）
rect_real = np.array([[-0.05,-0.01], [0.2,-0.01], [0.2,-0.05], [-0.05,-0.05]])
poly_rect = Polygon(rect_real)

intersection_area = poly_contour.intersection(poly_rect).area
print(f"实际相交面积：{intersection_area*1e4}(cm^2)")


# ---开始绘图---
plt.figure(figsize=(10, 8))

# 1. 绘制原始点云
plt.scatter(points[:, 0], points[:, 1], s=2, color='blue', alpha=0.5, label='Pointcloud')

# 2. 绘制轮廓（红色）
contour_plot = np.vstack([contour_real, contour_real[0]])  # 闭合轮廓
plt.plot(contour_plot[:,0], contour_plot[:,1], color='red', linewidth=2, label='Contour')

# 3. 绘制测试点
point_color = 'green' if is_inside else 'red'
plt.scatter(*test_point, color=point_color, s=100, marker='*', label='Test TCP Point')

# 4. 绘制矩形
rect_plot = np.vstack([rect_real, rect_real[0]])  # 闭合矩形
plt.plot(rect_plot[:,0], rect_plot[:,1], color='purple', linewidth=2, linestyle='--', label='Test Gripper Bounding Box')

# 额外标注面积结果
centroid = poly_rect.centroid.coords[0]
plt.text(centroid[0], centroid[1], f'Area={(intersection_area*1e4):.2f}', color='purple', fontsize=12, ha='center')

# 设置其他显示选项
plt.title('Pointcloud, Contour, Test TCP Point, and Test Gripper Bounding Box')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='best')
plt.axis('equal')
plt.grid(True)

plt.show()

