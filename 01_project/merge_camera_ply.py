#!/usr/bin/env python3
"""
提取物体（通过背景减除）并重建模型  — 调试增强版

功能：
- 将一批背景点云合并成背景模型（体素下采样）
- 对包含物体的点云，按“到背景最近距离”阈值剔除背景点
- 输出每帧前景（物体）点云，合并并重建网格
- 详细日志 + 可选前N帧可视化（过滤前/后对比）

使用：直接运行该脚本（无需命令行参数）
依赖：open3d, numpy
"""

import os
import glob
import time
import numpy as np
import open3d as o3d

# ====== 路径配置 ======
BACKGROUND_DIR = "Object/Verification_examples/ply CubeSat/V3/background"
FOREGROUND_DIR = "Object/Verification_examples/ply CubeSat/V3"
OUTPUT_DIR = os.path.join(FOREGROUND_DIR, "extracted_object")

# ====== 参数 ======
VOXEL_SIZE = 0.005           # 背景合并与物体合并时的下采样体素
DIST_THRESHOLD = 0.01        # 与背景的最小区分距离（米）；越大，前景越干净，但可能丢细节
MIN_POINTS_PER_FRAME = 50     # 一帧提取后少于该点数则跳过
RECONSTRUCT = True           # 是否进行网格重建
POISSON_DEPTH = 10

# 可选：前 N 帧做可视化（帮助调参）。0 表示不弹窗。
VISUALIZE_FIRST_N = 0


def log(msg):
    print(f"[INFO] {msg}")


def read_pointclouds_from(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.ply")))
    pcs = []
    for f in files:
        pc = o3d.io.read_point_cloud(f)
        pc.remove_non_finite_points()
        pcs.append(pc)
    return files, pcs


def merge_pointclouds(pcds, voxel_size=None):
    merged = o3d.geometry.PointCloud()
    for p in pcds:
        merged += p
    if voxel_size and len(merged.points) > 0:
        merged = merged.voxel_down_sample(voxel_size)
    return merged


def subtract_background(fg: o3d.geometry.PointCloud, bg: o3d.geometry.PointCloud, threshold=0.01):
    """返回：前景点云中，与背景最近距离 > threshold 的点（即保留非背景）。
    ⚠️ 注意方向：应使用 fg.compute_point_cloud_distance(bg)！
    """
    if len(fg.points) == 0 or len(bg.points) == 0:
        return o3d.geometry.PointCloud()
    dists = fg.compute_point_cloud_distance(bg)  # 对每个前景点，找背景最近邻距离
    dists = np.asarray(dists)
    keep_mask = dists > threshold
    idx = np.where(keep_mask)[0]
    if idx.size == 0:
        return o3d.geometry.PointCloud()
    return fg.select_by_index(idx)


def preprocess_for_reconstruction(pcd: o3d.geometry.PointCloud):
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=50))
        pcd.orient_normals_consistent_tangent_plane(50)
    return pcd


def reconstruct_mesh(pcd: o3d.geometry.PointCloud):
    pcd = preprocess_for_reconstruction(pcd)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=POISSON_DEPTH)
    densities = np.asarray(densities)
    keep = densities > np.percentile(densities, 5)
    mesh.remove_vertices_by_mask(~keep)
    mesh.compute_vertex_normals()
    return mesh


def main():
    t0 = time.time()
    # 1) 读取并合并背景
    log("读取背景点云...")
    bg_files, bg_pcds = read_pointclouds_from(BACKGROUND_DIR)
    if not bg_files:
        raise FileNotFoundError(f"背景目录无 .ply：{BACKGROUND_DIR}")
    background = merge_pointclouds(bg_pcds, voxel_size=VOXEL_SIZE)
    log(f"背景点云合并完成：文件 {len(bg_files)} 个，下采样后点数 {len(background.points)}")

    # 2) 读取前景
    log("读取前景点云...")
    fg_files = sorted(glob.glob(os.path.join(FOREGROUND_DIR, "*.ply")))
    if not fg_files:
        raise FileNotFoundError(f"未找到任何前景 .ply：{FOREGROUND_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    object_frames = []

    # 3) 逐帧背景减除
    for i, path in enumerate(fg_files):
        name = os.path.basename(path)
        log(f"处理 {name} ({i+1}/{len(fg_files)}) ...")
        fg = o3d.io.read_point_cloud(path)
        fg.remove_non_finite_points()
        n_before = len(fg.points)
        if n_before == 0:
            log(f"⚠️ {name} 原始为空，跳过")
            continue

        # 可选可视化：原始前景
        if i < VISUALIZE_FIRST_N:
            o3d.visualization.draw_geometries([fg], window_name=f"原始前景: {name}")

        # 背景减除
        t_sub = time.time()
        obj = subtract_background(fg, background, threshold=DIST_THRESHOLD)
        dt = time.time() - t_sub
        n_after = len(obj.points)
        log(f"  减除完成：前景点 {n_before} → 物体点 {n_after}，耗时 {dt:.2f}s")

        if n_after < MIN_POINTS_PER_FRAME:
            log(f"  ⚠️ {name} 物体点过少(<{MIN_POINTS_PER_FRAME})，已跳过")
            continue

        # 可选可视化：提取后的物体
        if i < VISUALIZE_FIRST_N:
            o3d.visualization.draw_geometries([obj], window_name=f"提取后物体: {name}")

        save_path = os.path.join(OUTPUT_DIR, f"object_{i:03d}.ply")
        o3d.io.write_point_cloud(save_path, obj)
        log(f"  ✅ 已保存: {save_path}")
        object_frames.append(obj)

    if len(object_frames) == 0:
        log("❌ 未能提取到任何有效前景物体。建议：增大 DIST_THRESHOLD 或检查前景与背景对齐情况。")
        return

    # 4) 合并全部物体点
    merged = merge_pointclouds(object_frames, voxel_size=VOXEL_SIZE)
    merged_path = os.path.join(OUTPUT_DIR, "merged_object.ply")
    o3d.io.write_point_cloud(merged_path, merged)
    log(f"✅ 已保存合并物体点云：{merged_path}（点数 {len(merged.points)}）")

    # 5) （可选）重建网格
    if RECONSTRUCT:
        mesh = reconstruct_mesh(merged)
        mesh_path = os.path.join(OUTPUT_DIR, "reconstructed_mesh.obj")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        log(f"✅ 已保存物体网格模型：{mesh_path}")
        if VISUALIZE_FIRST_N:
            o3d.visualization.draw_geometries([mesh], window_name="物体网格")

    # 6) 显示合并点云
    o3d.visualization.draw_geometries([merged], window_name="物体点云（合并）")
    log(f"全部完成，用时 {time.time()-t0:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"运行出错：{e}")
        import sys
        sys.exit(1)
