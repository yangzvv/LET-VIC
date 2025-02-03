import os
import open3d as o3d
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from pypcd.pypcd import PointCloud
current_folder_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(f'{current_folder_path}/../..')
curDirectory = os.getcwd()
print(curDirectory)

def convert_pcd_bin_to_pcd(input_bin_path, output_pcd_path, fields=None):
    """
    将 .pcd.bin 文件转换为标准 .pcd 文件。
    
    Args:
        input_bin_path (str): 输入的 .pcd.bin 文件路径。
        output_pcd_path (str): 输出的 .pcd 文件路径。
        fields (list): 点云字段名称（默认 ['x', 'y', 'z', 'intensity']）。
    """
    if fields is None:
        fields = ['x', 'y', 'z', 'intensity']  # 默认点云字段

    # 根据字段数确定点云的维度
    num_fields = len(fields)

    # 从 .pcd.bin 文件中读取二进制数据
    try:
        data = np.fromfile(input_bin_path, dtype=np.float32).reshape(-1, num_fields)
        print(f"Loaded point cloud with shape: {data.shape}")
    except Exception as e:
        print(f"Error reading .pcd.bin file: {e}")
        return

    # 构造 open3d 的点云对象
    pcd = o3d.geometry.PointCloud()

    # 设置点的坐标
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])

    # 如果有强度信息（或其他属性）
    if num_fields > 3:
        # 假设强度位于第 4 列（index 3）
        intensity = data[:, 3]
        colors = np.tile(intensity[:, None], (1, 3))  # 将强度映射为灰度颜色
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # 保存为标准的 .pcd 文件
    o3d.io.write_point_cloud(output_pcd_path, pcd)
    print(f"Point cloud saved to: {output_pcd_path}")


def batch_convert_pcd_bin_to_pcd(input_dir, output_dir, fields=None):
    """
    批量处理文件夹中的 .pcd.bin 文件并转换为标准的 .pcd 文件。
    
    Args:
        input_dir (str): 包含 .pcd.bin 文件的输入文件夹路径。
        output_dir (str): 转换后的 .pcd 文件输出文件夹路径。
        fields (list): 点云字段名称（默认 ['x', 'y', 'z', 'intensity']）。
    """
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_dir):
        # 检查文件是否以 .pcd.bin 结尾
        if file_name.endswith('.pcd.bin'):
            input_file_path = os.path.join(input_dir, file_name)
            output_file_name = file_name.replace('.pcd.bin', '.pcd')
            output_file_path = os.path.join(output_dir, output_file_name)

            # 转换文件
            print(f"Processing {file_name}...")
            convert_pcd_bin_to_pcd(input_file_path, output_file_path, fields)


def process_nuscenes_pcd_bin(lidar_path, output_path):
    """
    使用 nuScenes devkit 读取 .pcd.bin 文件并转换为标准 .pcd 文件。
    
    Args:
        nusc (NuScenes): nuScenes 数据集对象。
        lidar_path (str): .pcd.bin 文件路径。
        output_path (str): 转换后 .pcd 文件的保存路径。
    """
    # 读取点云
    point_cloud = LidarPointCloud.from_file(lidar_path)
    
    # 提取点云数据
    points = point_cloud.points.T  # 转置，获取 [N, 4] 格式 (x, y, z, intensity)
    
    # 打印点云数据形状
    print(f"Loaded point cloud with shape: {points.shape}")
    
    # 构造 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 设置坐标

    # 如果存在强度信息，将其映射为灰度颜色
    if points.shape[1] > 3:
        intensity = points[:, 3]
        colors = np.tile(intensity[:, None], (1, 3))  # 将强度映射为颜色
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存为 .pcd 文件
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud saved to: {output_path}")


def process_nuscenes_pcd_bin_with_pypcd(lidar_path, output_path):
    """
    使用 nuScenes devkit 读取 .pcd.bin 文件并保存为包含 intensity 的标准 .pcd 文件。
    
    Args:
        lidar_path (str): 输入的 .pcd.bin 文件路径。
        output_path (str): 输出的 .pcd 文件路径。
    """
    # 读取点云数据
    point_cloud = LidarPointCloud.from_file(lidar_path)
    points = point_cloud.points.T  # 转置为 [N, 4] 格式

    # 构建 PCD 文件的头部
    num_points = points.shape[0]
    header = {
        'version': .7,
        'fields': ['x', 'y', 'z', 'intensity'],
        'size': [4, 4, 4, 4],  # 每个字段的大小 (float32)
        'type': ['F', 'F', 'F', 'F'],  # 数据类型
        'count': [1, 1, 1, 1],  # 每个点的分量数量
        'width': num_points,  # 点的数量
        'height': 1,  # 1 表示无组织点云
        'viewpoint': [0, 0, 0, 1, 0, 0, 0],  # 默认视点
        'points': num_points,  # 点的数量
        'data': 'ascii'  # 或 'binary'，ASCII 更容易调试
    }

    # 构建 PCD 数据
    pc_data = np.core.records.fromarrays(
        points.T,  # 转置回 [4, N] 格式
        names=header['fields'],  # 字段名称
        formats=['f4', 'f4', 'f4', 'f4']  # 对应字段类型
    )
    pcd = PointCloud(header, pc_data)

    # 保存为 .pcd 文件
    pcd.save_pcd(output_path, compression='ascii')
    print(f"Point cloud with intensity saved to: {output_path}")


def process_all_pcd_bin_in_folder(input_folder, output_folder):
    """
    批量处理文件夹中的 .pcd.bin 文件，转换为 .pcd 文件。
    
    Args:
        nusc (NuScenes): nuScenes 数据集对象。
        input_folder (str): 包含 .pcd.bin 文件的文件夹路径。
        output_folder (str): 转换后 .pcd 文件的保存文件夹。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.pcd.bin'):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name.replace('.pcd.bin', '.pcd'))
            print(f"Processing {input_file} ...")
            process_nuscenes_pcd_bin_with_pypcd(input_file, output_file)


if __name__ == '__main__':
    # 设置输入和输出文件夹
    input_pcdbin_dir = "/home/yangzhenwei/datasets/V2X-Sim-2.0/sweeps/LIDAR_TOP_id_0"
    output_pcd_dir = "/home/yangzhenwei/datasets/V2X-Sim-2.0/sweeps/LIDAR_TOP_id_0"

    # 批量转换
    process_all_pcd_bin_in_folder(input_pcdbin_dir, output_pcd_dir)
