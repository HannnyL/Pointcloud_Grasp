import open3d as o3d   
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

pcd = o3d.io.read_point_cloud(r"D:\Codecouldcode\099.MA_Hanyu\Object\Verification_examples\01_Bottom_CubeSat_sampled.pcd")
scale_factor = 1/1000
pcd.scale(scale_factor, pcd.get_center())
dists = pcd.compute_nearest_neighbor_distance()
avg_d  = np.mean(dists)

logging.info(f"Avg NN distance: {avg_d:.4f}")