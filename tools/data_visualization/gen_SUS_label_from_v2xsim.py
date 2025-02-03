import os
import json
import math
import pickle
import numpy as np
import open3d as o3d
from pypcd.pypcd import PointCloud
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from utils import *
from pyquaternion import Quaternion
from mmdet3d.core.bbox import LiDARInstance3DBoxes
current_folder_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(f'{current_folder_path}/../..')


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


def process_nuscenes_pcd_bin_with_pypcd(lidar_path, output_path):
    point_cloud = LidarPointCloud.from_file(lidar_path)
    points = point_cloud.points.T 
    num_points = points.shape[0]
    header = {
        'version': .7,
        'fields': ['x', 'y', 'z', 'intensity'],
        'size': [4, 4, 4, 4], 
        'type': ['F', 'F', 'F', 'F'],  
        'count': [1, 1, 1, 1], 
        'width': num_points, 
        'height': 1, 
        'viewpoint': [0, 0, 0, 1, 0, 0, 0], 
        'points': num_points, 
        'data': 'ascii'
    }

    pc_data = np.core.records.fromarrays(
        points.T,
        names=header['fields'],
        formats=['f4', 'f4', 'f4', 'f4']
    )
    pcd = PointCloud(header, pc_data)
    pcd.save_pcd(output_path, compression='ascii')


def transform_box(center, dims, rotation_z, R, T):
    center = np.array(center).reshape(3, 1) 
    T = np.array(T).reshape(3, 1) 
    transformed_center = (R @ center + T).flatten()
    delta_rotation_z = np.arctan2(R[1, 0], R[0, 0]) 
    transformed_rotation_z = rotation_z + delta_rotation_z

    return transformed_center[0], transformed_center[1], transformed_center[2], dims[0], dims[1], dims[2], transformed_rotation_z


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


def gen_sus_label_from_v2xsim_nuscenes_results(nuscenes_dataset_path, let_vic_result_file_path, output_path):
    os.system(f"rm -r {output_path}")
    os.makedirs(output_path, exist_ok=True)
    sequence_data = load_pkl(let_vic_result_file_path)["bbox_results"]
    nusc = NuScenes(version="v1.0-trainval", dataroot=nuscenes_dataset_path)

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
        camera_data = nusc.get('sample_data', sample_info['data']['CAM_FRONT_id_1'])
        lidar_path = nusc.get_sample_data_path(lidar_data['token'])
        camera_path = nusc.get_sample_data_path(camera_data['token'])
        frame_id = lidar_path.split('/')[-1].split('.')[0]
        os.system(f"cp {camera_path} {output_seq_path}/camera/front/{frame_id}.jpg")
        process_nuscenes_pcd_bin_with_pypcd(lidar_path, f"{output_seq_path}/lidar/{frame_id}.pcd")

        output_label_file_path = f'{output_seq_path}/label/{frame_id}.json'

        list_data = frame_data["boxes_3d_det"]
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


def gen_sus_label_from_v2xsim_nuscenes_infos(nuscenes_dataset_path, let_vic_infos_file_path, output_path, cam="CAM_FRONT_id_1", lidar="LIDAR_TOP"):
    os.system(f"rm -r {output_path}")
    os.makedirs(output_path, exist_ok=True)
    sequence_data = load_pkl(let_vic_infos_file_path)["infos"]
    nusc = NuScenes(version="v1.0-trainval", dataroot=nuscenes_dataset_path)

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
        camera_data = nusc.get('sample_data', sample_info['data'][cam])
        lidar_path = nusc.get_sample_data_path(lidar_data['token'])
        camera_path = nusc.get_sample_data_path(camera_data['token'])
        frame_id = lidar_path.split('/')[-1].split('.')[0]
        os.system(f"cp {camera_path} {output_seq_path}/camera/front/{frame_id}.jpg")
        process_nuscenes_pcd_bin_with_pypcd(lidar_path, f"{output_seq_path}/lidar/{frame_id}.pcd")

        output_label_file_path = f'{output_seq_path}/label/{frame_id}.json'

        list_data = frame_data["gt_boxes"]
        list_sus_label_info = []
        for i in range(len(list_data)):
            if frame_data["valid_flag"][i]:
                sus_label_data = {}
                sus_label_data["obj_id"] = str(frame_data["gt_inds"][i])
                sus_label_data["obj_type"] = str(frame_data["gt_names"][i])
                sus_label_data["psr"] = {}
                sus_label_data["psr"]["position"] = {}
                sus_label_data["psr"]["position"]["x"] = float(list_data[i, 0])
                sus_label_data["psr"]["position"]["y"] = float(list_data[i, 1])
                sus_label_data["psr"]["position"]["z"] = float(list_data[i, 2])
                sus_label_data["psr"]["rotation"] = {}
                sus_label_data["psr"]["rotation"]["x"] = 0.0
                sus_label_data["psr"]["rotation"]["y"] = 0.0
                sus_label_data["psr"]["rotation"]["z"] = float(- list_data[i, 6] - np.pi / 2)
                sus_label_data["psr"]["scale"] = {}
                sus_label_data["psr"]["scale"]["x"] = float(list_data[i, 4])
                sus_label_data["psr"]["scale"]["y"] = float(list_data[i, 3])
                sus_label_data["psr"]["scale"]["z"] = float(list_data[i, 5])
                list_sus_label_info.append(sus_label_data)
        write_json(output_label_file_path, list_sus_label_info)


def gen_sus_label_from_v2xsim_nuscenes_infos_inf2veh(nuscenes_dataset_path, let_vic_infos_file_path_a, let_vic_infos_file_path_b, output_path, cam="CAM_FRONT_id_1", lidar="LIDAR_TOP"):
    os.system(f"rm -r {output_path}")
    os.makedirs(output_path, exist_ok=True)
    sequence_data_veh = load_pkl(let_vic_infos_file_path_a)["infos"]
    dict_sequence_data_veh = {}
    for m in sequence_data_veh:
        k = m["lidar_path"].split("/")[-1].split(".")[0]
        if k in dict_sequence_data_veh:
            print(f'{k} in dict_sequence_data_veh keys')
        else:
            dict_sequence_data_veh[k] = m
    sequence_data_inf = load_pkl(let_vic_infos_file_path_b)["infos"]
    dict_sequence_data_inf = {}
    for m in sequence_data_inf:
        k = m["lidar_path"].split("/")[-1].split(".")[0]
        if k in dict_sequence_data_inf:
            print(f'{k} in dict_sequence_data_inf keys')
        else:
            dict_sequence_data_inf[k] = m
    nusc = NuScenes(version="v1.0-trainval", dataroot=nuscenes_dataset_path)

    for key_i, frame_i in tqdm(dict_sequence_data_veh.items()):
        sample_token = frame_i["token"]
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
        camera_data = nusc.get('sample_data', sample_info['data'][cam])
        lidar_path = nusc.get_sample_data_path(lidar_data['token'])
        camera_path = nusc.get_sample_data_path(camera_data['token'])
        os.system(f"cp {camera_path} {output_seq_path}/camera/front/{key_i}.jpg")
        process_nuscenes_pcd_bin_with_pypcd(lidar_path, f"{output_seq_path}/lidar/{key_i}.pcd")

        output_label_file_path = f'{output_seq_path}/label/{key_i}.json'

        list_data = dict_sequence_data_inf[key_i]["gt_boxes"]
        list_sus_label_info = []
        for i in range(len(list_data)):
            if dict_sequence_data_inf[key_i]["valid_flag"][i]:
                sus_label_data = {}

                veh_l2e_r = np.array(Quaternion(dict_sequence_data_veh[key_i]["lidar2ego_rotation"]).rotation_matrix)
                veh_l2e_t = np.array(dict_sequence_data_veh[key_i]["lidar2ego_translation"]).reshape(3)
                veh_e2g_r = np.array(Quaternion(dict_sequence_data_veh[key_i]["ego2global_rotation"]).rotation_matrix)
                veh_e2g_t = np.array(dict_sequence_data_veh[key_i]["ego2global_translation"]).reshape(3)

                inf_l2e_r = np.array(Quaternion(dict_sequence_data_inf[key_i]["lidar2ego_rotation"]).rotation_matrix)
                inf_l2e_t = np.array(dict_sequence_data_inf[key_i]["lidar2ego_translation"]).reshape(3)
                inf_e2g_r = np.array(Quaternion(dict_sequence_data_inf[key_i]["ego2global_rotation"]).rotation_matrix)
                inf_e2g_t = np.array(dict_sequence_data_inf[key_i]["ego2global_translation"]).reshape(3)

                lidar_veh2inf_r = ((veh_l2e_r.T @ veh_e2g_r.T) @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T)).T
                lidar_veh2inf_t = (veh_l2e_t @ veh_e2g_r.T + veh_e2g_t) @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T)
                lidar_veh2inf_t -= (inf_e2g_t @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T) + inf_l2e_t @ (np.linalg.inv(inf_l2e_r).T))

                lidar_inf2veh_r = reverse_matrix(lidar_veh2inf_r)
                lidar_inf2veh_t = - np.dot(lidar_inf2veh_r, lidar_veh2inf_t)

                inf_lidar_location = [float(list_data[i, 0]), float(list_data[i, 1]), float(list_data[i, 2])]
                inf_lidar_dimensions = [float(list_data[i, 4]), float(list_data[i, 3]), float(list_data[i, 5])]
                inf_rotation_z = float(- list_data[i, 6] - np.pi / 2)

                # method 1
                inf_lidar_8points = get_lidar_3d_8points(inf_lidar_dimensions, inf_lidar_location, inf_rotation_z)
                veh_lidar_8points = (lidar_inf2veh_r @ np.array(inf_lidar_8points).T + lidar_inf2veh_t.reshape(3, 1)).T.tolist()
                veh_x, veh_y, veh_z, veh_l, veh_w, veh_h, veh_lidar_rotation = get_7points_from_8points(veh_lidar_8points)

                # method 2
                # veh_x, veh_y, veh_z, veh_l, veh_w, veh_h, veh_lidar_rotation = transform_box(inf_lidar_location, inf_lidar_dimensions, inf_rotation_z, lidar_inf2veh_r, lidar_inf2veh_t)

                sus_label_data["obj_id"] = str(dict_sequence_data_inf[key_i]["gt_inds"][i])
                sus_label_data["obj_type"] = str(dict_sequence_data_inf[key_i]["gt_names"][i])
                sus_label_data["psr"] = {}
                sus_label_data["psr"]["position"] = {}
                sus_label_data["psr"]["position"]["x"] = veh_x
                sus_label_data["psr"]["position"]["y"] = veh_y
                sus_label_data["psr"]["position"]["z"] = veh_z
                sus_label_data["psr"]["rotation"] = {}
                sus_label_data["psr"]["rotation"]["x"] = 0.0
                sus_label_data["psr"]["rotation"]["y"] = 0.0
                sus_label_data["psr"]["rotation"]["z"] = veh_lidar_rotation
                sus_label_data["psr"]["scale"] = {}
                sus_label_data["psr"]["scale"]["x"] = veh_l
                sus_label_data["psr"]["scale"]["y"] = veh_w
                sus_label_data["psr"]["scale"]["z"] = veh_h
                # print(sus_label_data)
                list_sus_label_info.append(sus_label_data)
        write_json(output_label_file_path, list_sus_label_info)


def gen_sus_label_from_v2xsim_spd_infos(spd_dataset_path, let_vic_infos_file_path, output_path):
    """
        Args:
            spd_dataset_path: "datasets/V2X-Seq-SPD-10Hz-O/vehicle-side"
            let_vic_infos_file_path: "data/infos/V2X-Seq-SPD-10Hz-O/vehicle-side/spd_infos_temporal_val.pkl"
            output_path: "/data/kongzhi/yangzhenwei/github-projects/SUSTechPOINTS/SPD-veh-infos"
        Returns:
            None
    """
    os.system(f"rm -r {output_path}")
    os.makedirs(output_path, exist_ok=True)
    sequence_data = load_pkl(let_vic_infos_file_path)["infos"]

    data_info = read_json(f'{spd_dataset_path}/data_info.json')
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
        os.system(f"cp {spd_dataset_path}/image/{frame_id}.jpg {output_seq_path}/camera/front/")
        os.system(f"cp {spd_dataset_path}/velodyne/{frame_id}.pcd {output_seq_path}/lidar/")

        output_label_file_path = f'{output_seq_path}/label/{frame_id}.json'

        list_data = frame_data["gt_boxes"]
        list_sus_label_info = []
        for i in range(len(list_data)):
            sus_label_data = {}
            sus_label_data["obj_id"] = str(frame_data["gt_inds"][i])
            sus_label_data["obj_type"] = str(frame_data["gt_names"][i])
            sus_label_data["psr"] = {}
            sus_label_data["psr"]["position"] = {}
            sus_label_data["psr"]["position"]["x"] = float(list_data[i, 0])
            sus_label_data["psr"]["position"]["y"] = float(list_data[i, 1])
            sus_label_data["psr"]["position"]["z"] = float(list_data[i, 2])
            sus_label_data["psr"]["rotation"] = {}
            sus_label_data["psr"]["rotation"]["x"] = 0.0
            sus_label_data["psr"]["rotation"]["y"] = 0.0
            sus_label_data["psr"]["rotation"]["z"] = float(- list_data[i, 6] - np.pi / 2)
            sus_label_data["psr"]["scale"] = {}
            sus_label_data["psr"]["scale"]["x"] = float(list_data[i, 4])
            sus_label_data["psr"]["scale"]["y"] = float(list_data[i, 3])
            sus_label_data["psr"]["scale"]["z"] = float(list_data[i, 5])
            list_sus_label_info.append(sus_label_data)
        write_json(output_label_file_path, list_sus_label_info)


if __name__ == '__main__':
    pass
