import open3d as o3d
import numpy as np


#todo 功能1 - 读取并显示.pcd或.ply文件
def read_show_pcd_or_ply():
# 替换路径
# path = r"D://Codecouldcode//099.MA_Hanyu//YCB_Dataset//002_master_chef_can.ply"
# 
    print("loadding -> 功能1：读取并显示.pcd或.ply文件")

    path = input("请输入文件路径：").replace("\\", "/")

    pcd = o3d.io.read_point_cloud(path)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.006, origin=[0, 0, 0]) #添加坐标轴
    o3d.visualization.draw_geometries([pcd,axis])



#todo 功能2 - 从模型采样点云
def sample_and_save_points_from_mesh():
    # "D:\\Codecouldcode\\099.MA_Hanyu\\Data_Example\\model_v1.2.ply"

    print("loadding -> 功能2：从模型采样点云")

    path = input("请输入文件路径：").replace("\\", "/")
    number_of_points = int(input("请输入采样点数：")) if input("默认采样点数10000，是否修改？（y/n）") == "y" else 10000
    
    mesh = o3d.io.read_triangle_mesh(path)
    print(f'原模型是否着色：{mesh.has_vertex_colors()}') 
    pcd = mesh.sample_points_poisson_disk(number_of_points)
    o3d.visualization.draw_geometries([pcd])

    print("保存文件中...")
    o3d.io.write_point_cloud(path.replace(".ply", "_sampled.pcd"), pcd)
    print("保存成功！")

# todo 功能3 - 创建并显示边界框
def create_and_show_bounding_box():

    print("loadding -> 功能3：创建并显示边界框")

    path = input("请输入文件路径：").replace("\\", "/")
    print("\nX轴-红色，Y轴-绿色，Z轴-蓝色|最小边界框OBB-红色，沿坐标轴边界框AABB-绿色\n")

    # 读取点云
    pcd = o3d.io.read_point_cloud(path)  # 替换成你的路径
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)

    # 获取 OBB
    obb = pcd.get_oriented_bounding_box()
    obb.color = [1, 0, 0]  # 红色盒子

    # 获取 AABB
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = [0, 1, 0]  # 绿色

    # 主轴方向
    R = obb.R
    center = obb.center
    x_axis, y_axis, z_axis = R[:, 0], R[:, 1], R[:, 2]

    # 可视化箭头（表示主轴）
    def make_arrow(start, direction, color):
        end = start + direction * 0.05
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([start, end]),
            lines=o3d.utility.Vector2iVector([[0, 1]]),
        )
        line.colors = o3d.utility.Vector3dVector([color])
        return line

    arrow_x = make_arrow(center, x_axis, [1, 0, 0])  # 红色
    arrow_y = make_arrow(center, y_axis, [0, 1, 0])  # 绿色
    arrow_z = make_arrow(center, z_axis, [0, 0, 1])  # 蓝色

    o3d.visualization.draw_geometries([pcd, obb, aabb, arrow_x, arrow_y, arrow_z])




functions = {
    "1":read_show_pcd_or_ply, 
    "2":sample_and_save_points_from_mesh,
    "3":create_and_show_bounding_box

    }

if __name__ == '__main__':

    while True:
        choice = input(
'''
****************************
请输入功能序号：
1、读取并显示.pcd或.ply文件 - Read and Show .pcd or .ply File；
2、从模型采样点云 - Sample Points from Mesh；
3、创建并显示边界框 - Create and Show Bounding Box；
-----------
0、退出程序。
****************************
'''
        )
        if choice == "0":
            break
        else:
            try:
                functions.get(choice, lambda: print("输入有误，请重新输入！"))()
                print("\n功能{}已执行完毕！".format(choice))
            except Exception as e:
                print(f"\n执行出错：{e}\n")

    print("程序已退出！")