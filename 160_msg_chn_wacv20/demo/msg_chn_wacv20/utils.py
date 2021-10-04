import numpy as np
import cv2

def make_depth_sparse(depth_map, depth_percentage):

    i, j = np.indices(depth_map.shape)
    coordinates = np.hstack((i.reshape((-1,1)), j.reshape((-1,1))))
    total_pixels = coordinates.shape[0]

    random_coordinate_ids = np.random.permutation(total_pixels)
    selected_pixels = coordinates[random_coordinate_ids[:int(total_pixels*(100-depth_percentage)/100)]]

    sparse_depth = depth_map.copy()
    sparse_depth[selected_pixels[:,0],selected_pixels[:,1]] = 0

    return sparse_depth

def draw_depth(depth_map, max_dist):

    norm_depth_map = 255*(1-depth_map/max_dist)
    norm_depth_map[norm_depth_map < 0] =0
    norm_depth_map[depth_map == 0] =0

    return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_MAGMA)

def update_depth_density(depth_density, depth_density_rate, min_density=0.5, max_density=10):

    if depth_density <= min_density or depth_density >= max_density:
        depth_density_rate = -depth_density_rate

    depth_density += depth_density_rate

    return depth_density, depth_density_rate