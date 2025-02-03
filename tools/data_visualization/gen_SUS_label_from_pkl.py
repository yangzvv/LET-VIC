import os
import json
import math
import pickle
import numpy as np
import open3d as o3d
from pypcd import pypcd
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from utils import *
from pyquaternion import Quaternion
from mmdet3d.core.bbox import LiDARInstance3DBoxes

type_id2str = {0: 'Trafficcone', 1: 'Pedestrian', 2: 'Car', 3: 'Cyclist', 4: 'Van', 5: 'Truck', 6: 'Bus', 7: 'Tricyclist',
               8: 'Motorcyclist', 9: 'Barrowlist'}
# type_str2id = dict([val, key] for key, val in type_id2str.items())
type_str2id = {'Trafficcone': 0, 'Pedestrian': 1, 'Car': 2, 'Cyclist': 3, 'Van': 4, 'Truck': 5, 'Bus': 6, 'Tricyclist': 7,
               'Motorcyclist': 8, 'Barrowlist': 9}


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def read_txt(path_txt):
    with open(path_txt, "r") as f:
        my_txt = f.readlines()
    return my_txt


def get_7points_from_8points(lidar_3d_8points):
    lidar_xy0, lidar_xy3, lidar_xy1 = lidar_3d_8points[0][0:2], lidar_3d_8points[3][0:2], lidar_3d_8points[1][0:2]
    lidar_z4, lidar_z0 = lidar_3d_8points[4][2], lidar_3d_8points[0][2]
    l = math.sqrt((lidar_xy0[0] - lidar_xy3[0]) ** 2 + (lidar_xy0[1] - lidar_xy3[1]) ** 2)
    w = math.sqrt((lidar_xy0[0] - lidar_xy1[0]) ** 2 + (lidar_xy0[1] - lidar_xy1[1]) ** 2)
    h = lidar_z4 - lidar_z0
    lidar_x0, lidar_y0 = lidar_3d_8points[0][0], lidar_3d_8points[0][1]
    lidar_x2, lidar_y2 = lidar_3d_8points[2][0], lidar_3d_8points[2][1]
    lidar_x = (lidar_x0 + lidar_x2) / 2
    lidar_y = (lidar_y0 + lidar_y2) / 2
    lidar_z = (lidar_z0 + lidar_z4) / 2

    lidar_rotation = get_label_lidar_rotation(lidar_3d_8points)
    return lidar_x, lidar_y, lidar_z, l, w, h, lidar_rotation


def trans_point_w2v(input_point, path_novatel2world, path_lidar2novatel, delta_x, delta_y):
    # world to novatel
    rotation, translation = get_novatel2world(path_novatel2world)
    new_rotation = reverse_matrix(rotation)
    new_translation = - np.dot(new_rotation, translation)
    point = trans_point(input_point, new_translation, new_rotation)

    # novatel to lidar
    rotation, translation = get_lidar2novatel(path_lidar2novatel)
    new_rotation = reverse_matrix(rotation)
    new_translation = - np.dot(new_rotation, translation)
    point = trans_point(point, new_translation, new_rotation)

    point = np.array(point).reshape(3, 1) + np.array([delta_x, delta_y, 0]).reshape(3, 1)
    point = point.reshape(1, 3).tolist()[0]

    return point


def gen_sus_label_from_univ2x_kitti_results(spd_dataset_path, vi, univ2x_result_file_path, output_path):
    os.system(f"rm -r {output_path}")
    sequence_data = load_pkl(univ2x_result_file_path)["bbox_results"]

    data_info = read_json(f'{spd_dataset_path}/{vi}/data_info.json')
    dict_frame_id2info = {}
    for i in tqdm(data_info):
        dict_frame_id2info[i["frame_id"]] = i

    for frame_data in tqdm(sequence_data):
        frame_id = frame_data["token"]
        frame_info = dict_frame_id2info[frame_id]
        output_seq_path = output_path + "/" + frame_info["sequence_id"]
        os.makedirs(output_seq_path + '/label', exist_ok=True)
        os.makedirs(output_seq_path + '/lidar', exist_ok=True)
        os.makedirs(output_seq_path + '/calib', exist_ok=True)
        os.makedirs(output_seq_path + '/camera/front', exist_ok=True)
        os.makedirs(output_seq_path + '/camera/left', exist_ok=True)
        os.makedirs(output_seq_path + '/camera/right', exist_ok=True)

        os.system(f"cp {spd_dataset_path}/{vi}/image/{frame_id}.jpg {output_seq_path}/camera/front/")
        os.system(f"cp {spd_dataset_path}/{vi}/velodyne/{frame_id}.pcd {output_seq_path}/lidar/")

        output_label_file_path = f'{output_seq_path}/label/{frame_id}.json'

        list_data = frame_data["boxes_3d"]
        list_sus_label_info = []
        for i in range(len(list_data)):
            sus_label_data = {}
            sus_label_data["obj_id"] = str(frame_data["track_ids"].numpy()[i])
            sus_label_data["obj_type"] = str(frame_data["labels_3d_det"].numpy()[i])
            sus_label_data["psr"] = {}
            sus_label_data["psr"]["position"] = {}
            sus_label_data["psr"]["position"]["x"] = float(list_data.gravity_center.numpy()[i, 0])
            sus_label_data["psr"]["position"]["y"] = float(list_data.gravity_center.numpy()[i, 1])
            sus_label_data["psr"]["position"]["z"] = float(list_data.gravity_center.numpy()[i, 2])
            sus_label_data["psr"]["rotation"] = {}
            sus_label_data["psr"]["rotation"]["x"] = 0.0
            sus_label_data["psr"]["rotation"]["y"] = 0.0
            sus_label_data["psr"]["rotation"]["z"] = float(- list_data.yaw.numpy()[i] - np.pi / 2)
            sus_label_data["psr"]["scale"] = {}
            sus_label_data["psr"]["scale"]["x"] = float(list_data.dims.numpy()[i, 1])
            sus_label_data["psr"]["scale"]["y"] = float(list_data.dims.numpy()[i, 0])
            sus_label_data["psr"]["scale"]["z"] = float(list_data.dims.numpy()[i, 2])
            list_sus_label_info.append(sus_label_data)
        write_json(output_label_file_path, list_sus_label_info)


def gen_sus_label_from_univ2x_nuscenes_results(nuscenes_dataset_path, univ2x_result_file_path, output_path):
    os.system(f"rm -r {output_path}")
    sequence_data = load_pkl(univ2x_result_file_path)
    nusc = NuScenes(version="v1.0-mini", dataroot=nuscenes_dataset_path)

    for frame_data in tqdm(sequence_data):
        sample_token = frame_data["token"]
        sample_info = nusc.get('sample', sample_token)
        timestamp = sample_info['timestamp']

        scene_id = nusc.get('scene', sample_info['scene_token'])['name']
        output_seq_path = output_path + "/" + scene_id
        os.makedirs(output_seq_path + '/label', exist_ok=True)
        os.makedirs(output_seq_path + '/lidar', exist_ok=True)
        os.makedirs(output_seq_path + '/calib', exist_ok=True)
        os.makedirs(output_seq_path + '/camera/front', exist_ok=True)
        os.makedirs(output_seq_path + '/camera/left', exist_ok=True)
        os.makedirs(output_seq_path + '/camera/right', exist_ok=True)

        lidar_data = nusc.get('sample_data', sample_info['data']['LIDAR_TOP'])
        camera_data = nusc.get('sample_data', sample_info['data']['CAM_FRONT'])
        lidar_path = nusc.get_sample_data_path(lidar_data['token'])
        camera_path = nusc.get_sample_data_path(camera_data['token'])
        os.system(f"cp {camera_path} {output_seq_path}/camera/front/{timestamp}.jpg")
        os.system(f"cp {lidar_path} {output_seq_path}/lidar/{timestamp}.bin")

        output_label_file_path = f'{output_seq_path}/label/{timestamp}.json'

        list_data = frame_data["boxes_3d"]
        list_sus_label_info = []
        for i in range(len(list_data)):
            sus_label_data = {}
            sus_label_data["obj_id"] = ""
            sus_label_data["obj_type"] = str(frame_data["labels_3d_det"].numpy()[i])
            sus_label_data["psr"] = {}
            sus_label_data["psr"]["position"] = {}
            sus_label_data["psr"]["position"]["x"] = float(list_data.gravity_center.numpy()[i, 0])
            sus_label_data["psr"]["position"]["y"] = float(list_data.gravity_center.numpy()[i, 1])
            sus_label_data["psr"]["position"]["z"] = float(list_data.gravity_center.numpy()[i, 2])
            sus_label_data["psr"]["rotation"] = {}
            sus_label_data["psr"]["rotation"]["x"] = 0.0
            sus_label_data["psr"]["rotation"]["y"] = 0.0
            sus_label_data["psr"]["rotation"]["z"] = float(- list_data.yaw.numpy()[i] - np.pi / 2)
            sus_label_data["psr"]["scale"] = {}
            sus_label_data["psr"]["scale"]["x"] = float(list_data.dims.numpy()[i, 1])
            sus_label_data["psr"]["scale"]["y"] = float(list_data.dims.numpy()[i, 0])
            sus_label_data["psr"]["scale"]["z"] = float(list_data.dims.numpy()[i, 2])
            list_sus_label_info.append(sus_label_data)
        write_json(output_label_file_path, list_sus_label_info)


def gen_sus_label_from_univ2x_dair_results(spd_dataset_path, vi, univ2x_anno_json_path, output_path):
    data_info = read_json(f'{spd_dataset_path}/{vi}/data_info.json')
    dict_frame_id2info = {}
    for i in tqdm(data_info):
        dict_frame_id2info[i["frame_id"]] = i

    anno_info = read_json(univ2x_anno_json_path)
    dict_frame_id2anno = {}
    for i in anno_info:
        if i["sample_token"] not in dict_frame_id2anno.keys():
            dict_frame_id2anno[i["sample_token"]] = []
        dict_frame_id2anno[i["sample_token"]].append(i)

    for frame_id in tqdm(dict_frame_id2anno.keys()):
        frame_info = dict_frame_id2info[frame_id]
        output_seq_path = output_path + "/" + frame_info["sequence_id"]
        os.makedirs(output_seq_path + '/label', exist_ok=True)
        os.makedirs(output_seq_path + '/lidar', exist_ok=True)
        os.makedirs(output_seq_path + '/calib', exist_ok=True)
        os.makedirs(output_seq_path + '/camera/front', exist_ok=True)
        os.makedirs(output_seq_path + '/camera/left', exist_ok=True)
        os.makedirs(output_seq_path + '/camera/right', exist_ok=True)

        os.system(f"cp {spd_dataset_path}/{vi}/image/{frame_id}.jpg {output_seq_path}/camera/front/")
        os.system(f"cp {spd_dataset_path}/{vi}/velodyne/{frame_id}.pcd {output_seq_path}/lidar/")

        output_label_file_path = f'{output_seq_path}/label/{frame_id}.json'

        novatel2world_path = f'{spd_dataset_path}/{vi}/calib/novatel_to_world/{frame_id}.json'
        lidar2novatel_path = f'{spd_dataset_path}/{vi}/calib/lidar_to_novatel/{frame_id}.json'
        delta_x = 0
        delta_y = 0

        frame_data = dict_frame_id2anno[frame_id]
        list_sus_label_info = []
        for i in frame_data:
            sus_label_data = {}
            w_label_3d_dimensions = [float(i["size"][1]), float(i["size"][0]), float(i["size"][2])]
            w_label_lidar_3d_location = [float(i["translation"][0]), float(i["translation"][1]), float(i["translation"][2])]
            w_label_lidar_rotation = Quaternion(i["rotation"]).yaw_pitch_roll[0]

            v_label_lidar_3d_location = trans_point_w2v(w_label_lidar_3d_location, novatel2world_path, lidar2novatel_path, delta_x, delta_y)
            list_w_lidar_3d_8_points = get_lidar_3d_8points(w_label_3d_dimensions, w_label_lidar_3d_location, w_label_lidar_rotation)
            list_v_lidar_3d_8_points = []
            for w_lidar_point in list_w_lidar_3d_8_points:
                v_lidar_point = trans_point_w2v(w_lidar_point, novatel2world_path, lidar2novatel_path, delta_x, delta_y)
                list_v_lidar_3d_8_points.append(v_lidar_point)
            v_label_lidar_rotation = get_label_lidar_rotation(list_v_lidar_3d_8_points)

            sus_label_data["obj_id"] = i['tracking_id']
            sus_label_data["obj_type"] = i['tracking_name']
            sus_label_data["psr"] = {}
            sus_label_data["psr"]["position"] = {}
            sus_label_data["psr"]["position"]["x"] = v_label_lidar_3d_location[0]
            sus_label_data["psr"]["position"]["y"] = v_label_lidar_3d_location[1]
            sus_label_data["psr"]["position"]["z"] = v_label_lidar_3d_location[2]
            sus_label_data["psr"]["rotation"] = {}
            sus_label_data["psr"]["rotation"]["x"] = 0.0
            sus_label_data["psr"]["rotation"]["y"] = 0.0
            sus_label_data["psr"]["rotation"]["z"] = float(v_label_lidar_rotation)
            sus_label_data["psr"]["scale"] = {}
            sus_label_data["psr"]["scale"]["x"] = w_label_3d_dimensions[0]
            sus_label_data["psr"]["scale"]["y"] = w_label_3d_dimensions[1]
            sus_label_data["psr"]["scale"]["z"] = w_label_3d_dimensions[2]
            list_sus_label_info.append(sus_label_data)
        write_json(output_label_file_path, list_sus_label_info)


def gen_sus_label_from_univ2x_results(spd_dataset_path, vi, results_json_path, output_path):
    data_info = read_json(f'{spd_dataset_path}/{vi}/data_info.json')
    dict_frame_id2info = {}
    for i in tqdm(data_info):
        dict_frame_id2info[i["frame_id"]] = i

    results_info = read_json(results_json_path)
    dict_frame_id2anno = results_info['results']

    output_seq_path = output_path
    os.makedirs(output_seq_path + '/label', exist_ok=True)
    os.makedirs(output_seq_path + '/lidar', exist_ok=True)
    os.makedirs(output_seq_path + '/calib', exist_ok=True)
    os.makedirs(output_seq_path + '/camera/front', exist_ok=True)
    os.makedirs(output_seq_path + '/camera/left', exist_ok=True)
    os.makedirs(output_seq_path + '/camera/right', exist_ok=True)

    for frame_id in tqdm(dict_frame_id2anno.keys()):
        os.system(f"cp {spd_dataset_path}/{vi}/image/{frame_id}.jpg {output_seq_path}/camera/front/")
        os.system(f"cp {spd_dataset_path}/{vi}/velodyne/{frame_id}.pcd {output_seq_path}/lidar/")

        output_label_file_path = f'{output_seq_path}/label/{frame_id}.json'

        if vi == "infrastructure-side":
            virtuallidar2world_path = f'{spd_dataset_path}/{vi}/calib/virtuallidar_to_world/{frame_id}.json'
            rotation_l2w, translation_l2w = get_virtuallidar2world(virtuallidar2world_path)
            rotation_w2l, translation_w2l = reverse(rotation_l2w, translation_l2w)
        else:
            lidar2novatel_path = f'{spd_dataset_path}/{vi}/calib/lidar_to_novatel/{frame_id}.json'
            novatel2world_path = f'{spd_dataset_path}/{vi}/calib/novatel_to_world/{frame_id}.json'
            rotation_l2e, translation_l2e = get_lidar2novatel(lidar2novatel_path)
            rotation_e2w, translation_e2w = get_novatel2world(novatel2world_path)
            rotation_e2l, translation_e2l = reverse(rotation_l2e, translation_l2e)
            rotation_w2e, translation_w2e = reverse(rotation_e2w, translation_e2w)
            rotation_w2l, translation_w2l = muilt_coord(rotation_w2e, translation_w2e, rotation_e2l, translation_e2l)


        delta_x = 0
        delta_y = 0

        frame_data = dict_frame_id2anno[frame_id]   #list
        for i in frame_data:
            sus_label_data = {}
            w_label_3d_dimensions = [float(i["size"][1]), float(i["size"][0]), float(i["size"][2])]
            w_label_lidar_3d_location = [float(i["translation"][0]), float(i["translation"][1]), float(i["translation"][2])]
            w_label_lidar_rotation = Quaternion(i["rotation"]).yaw_pitch_roll[0]

            v_label_lidar_3d_location = trans_point(w_label_lidar_3d_location, rotation_w2l, translation_w2l)
            list_w_lidar_3d_8_points = get_lidar_3d_8points(w_label_3d_dimensions, w_label_lidar_3d_location, w_label_lidar_rotation)
            list_v_lidar_3d_8_points = []
            for w_lidar_point in list_w_lidar_3d_8_points:
                v_lidar_point = trans_point(w_lidar_point, rotation_w2l, translation_w2l)
                list_v_lidar_3d_8_points.append(v_lidar_point)
            v_label_lidar_rotation = get_label_lidar_rotation(list_v_lidar_3d_8_points)


            sus_label_data["obj_id"] = i['tracking_id'] 
            sus_label_data["obj_type"] = i['tracking_name'] 
            sus_label_data["psr"] = {}
            sus_label_data["psr"]["position"] = {}
            sus_label_data["psr"]["position"]["x"] = v_label_lidar_3d_location[0]
            sus_label_data["psr"]["position"]["y"] = v_label_lidar_3d_location[1]
            sus_label_data["psr"]["position"]["z"] = v_label_lidar_3d_location[2]
            sus_label_data["psr"]["rotation"] = {}
            sus_label_data["psr"]["rotation"]["x"] = 0.0
            sus_label_data["psr"]["rotation"]["y"] = 0.0
            sus_label_data["psr"]["rotation"]["z"] = float(v_label_lidar_rotation)
            sus_label_data["psr"]["scale"] = {}
            sus_label_data["psr"]["scale"]["x"] = w_label_3d_dimensions[0]
            sus_label_data["psr"]["scale"]["y"] = w_label_3d_dimensions[1]
            sus_label_data["psr"]["scale"]["z"] = w_label_3d_dimensions[2]
            list_sus_label_info.append(sus_label_data)
        write_json(output_label_file_path, list_sus_label_info)


if __name__ == '__main__':
    pass
