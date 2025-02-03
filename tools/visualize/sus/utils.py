import json
import math
import numpy as np
import open3d as o3d
from pypcd import pypcd
import pickle

def read_pkl(path):
    with open(path,'rb') as fp:
        data = pickle.load(fp)
    return data

def read_json(path_json):
    with open(path_json, 'r') as load_f:
        data_json = json.load(load_f)
    return data_json


def write_json(path_json, data_json):
    with open(path_json, "w") as f:
        json.dump(data_json, f)


def read_pcd(path_pcd):
    pointpillar = o3d.io.read_point_cloud(path_pcd)
    points = np.asarray(pointpillar.points)
    points = points.tolist()
    return points


def show_pcd(path_pcd):
    pcd = read_pcd(path_pcd)
    o3d.visualization.draw_geometries([pcd])


def write_pcd(path_pcd, new_points, path_save):
    pc = pypcd.PointCloud.from_path(path_pcd)
    pc.pc_data['x'] = np.array([a[0] for a in new_points])
    pc.pc_data['y'] = np.array([a[1] for a in new_points])
    pc.pc_data['z'] = np.array([a[2] for a in new_points])
    pc.save_pcd(path_save, compression='binary_compressed')


def reverse_matrix(R):
    R = np.matrix(R)
    rev_R = R.I
    rev_R = np.array(rev_R)
    return rev_R


def muilt_coord(rotationA2B, translationA2B, rotationB2C, translationB2C):
    rotationA2B = np.array(rotationA2B).reshape(3, 3)
    rotationB2C = np.array(rotationB2C).reshape(3, 3)
    rotation = np.dot(rotationB2C, rotationA2B)
    translationA2B = np.array(translationA2B).reshape(3, 1)
    translationB2C = np.array(translationB2C).reshape(3, 1)
    translation = np.dot(rotationB2C, translationA2B) + translationB2C
    return rotation, translation


def reverse(rotation, translation):
    rev_rotation = reverse_matrix(rotation)
    rev_translation = -np.dot(rev_rotation, translation)
    return rev_rotation, rev_translation


def get_camera_3d_8points(label_3d_dimensions, camera_3d_location, rotation_y):
    camera_rotation = np.matrix(
        [
            [math.cos(rotation_y), 0, math.sin(rotation_y)],
            [0, 1, 0],
            [-math.sin(rotation_y), 0, math.cos(rotation_y)]
        ]
    )
    l, w, h = label_3d_dimensions
    corners_3d_camera = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [0, 0, 0, 0, -h, -h, -h, -h],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        ]
    )
    camera_3d_8points = camera_rotation * corners_3d_camera + np.matrix(camera_3d_location).T
    return camera_3d_8points.T.tolist()


def get_camera_3d_alpha_rotation(corners_3d_cam, center_in_cam):
    x0, z0 = corners_3d_cam[0][0], corners_3d_cam[0][2]
    x3, z3 = corners_3d_cam[3][0], corners_3d_cam[3][2]
    dx, dz = x0 - x3, z0 - z3
    rotation_y = -math.atan2(dz, dx) 
    alpha = rotation_y - (-math.atan2(-center_in_cam[2], -center_in_cam[0])) + math.pi / 2 
    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi
    return alpha, rotation_y


def get_lidar_3d_8points(label_3d_dimensions, lidar_3d_location, rotation_z):
    lidar_rotation = np.matrix(
        [
            [math.cos(rotation_z), -math.sin(rotation_z), 0],
            [math.sin(rotation_z), math.cos(rotation_z), 0],
            [0, 0, 1]
        ]
    )
    l, w, h = label_3d_dimensions
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )
    lidar_3d_8points = lidar_rotation * corners_3d_lidar + np.matrix(lidar_3d_location).T
    return lidar_3d_8points.T.tolist()


def get_label_lidar_rotation(lidar_3d_8_points):
    x0, y0 = lidar_3d_8_points[0][0], lidar_3d_8_points[0][1]
    x3, y3 = lidar_3d_8_points[3][0], lidar_3d_8_points[3][1]
    dx, dy = x0 - x3, y0 - y3
    rotation_z = math.atan2(dy, dx)
    return rotation_z


def get_rotation_translation(input_path):
    matrix_info = read_json(input_path)
    rotation = matrix_info['rotation']
    translation = matrix_info['translation']
    return rotation, translation


def get_virtuallidar2world(path_virtuallidar2world):
    virtuallidar2world = read_json(path_virtuallidar2world)
    rotation = virtuallidar2world['rotation']
    translation = virtuallidar2world['translation']
    return rotation, translation


def get_novatel2world(path_novatel2world):
    novatel2world = read_json(path_novatel2world)
    rotation = novatel2world['rotation']
    translation = novatel2world['translation']
    return rotation, translation


def get_lidar2novatel(path_lidar2novatel):
    lidar2novatel = read_json(path_lidar2novatel)
    rotation = lidar2novatel['transform']['rotation']
    translation = lidar2novatel['transform']['translation']
    return rotation, translation


def get_lidar2camera(path_lidar2camera):
    lidar2camera = read_json(path_lidar2camera)
    rotation = lidar2camera['rotation']
    translation = lidar2camera['translation']
    return rotation, translation


def get_cam_calib_intrinsic(calib_path):
    my_json = read_json(calib_path)
    cam_K = my_json["cam_K"]
    calib = np.zeros([3, 4])
    calib[:3, :3] = np.array(cam_K).reshape([3, 3], order="C")
    return calib


def trans_point(input_point, rotation, translation):
    input_point = np.array(input_point).reshape(3, 1)
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    output_point = np.dot(rotation, input_point).reshape(3, 1) + np.array(translation).reshape(3, 1)
    output_point = output_point.reshape(1, 3).tolist()
    return output_point[0]


def trans(input_point, rotation, translation):
    input_point = np.array(input_point).T.reshape(3, -1)
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    output_point = np.dot(rotation, input_point).reshape(3, -1) + np.array(translation).reshape(3, 1)
    return output_point.T.tolist()
