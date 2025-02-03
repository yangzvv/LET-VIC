from PIL import Image, ImageDraw, ImageFont
import os.path as osp
import json
import glob
from tqdm import tqdm
import cv2
import numpy as np

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)

    return data

def merge_images(img_inf_path, img_veh_path, img_output_path):
    W_1, H_1 = 640, 360
    W_2, H_2 = 640, 360
    W_3, H_3 = 640, 640

    image_inf = Image.open(img_inf_path)
    image_veh = Image.open(img_veh_path)
    image_output = Image.open(img_output_path)

    image_inf = image_inf.resize((W_1, H_1))
    image_veh = image_veh.resize((W_2, H_2))
    image_output = image_output.resize((W_3, H_3))


    merged_image = Image.new('RGB', (W_1 + W_3 + 20, H_1 + H_2 + 20), color=(255, 255, 255))
    
    merged_image.paste(image_inf, (0, 0))
    merged_image.paste(image_veh, (0, H_1 + 20))
    merged_image.paste(image_output, (W_1 + 20, 50))

    draw = ImageDraw.Draw(merged_image)
    font = ImageFont.truetype('./GidoleFont/Gidole-Regular.ttf', 30)
    textsize = 15
    text = 'Infrastructure Image'
    draw.text((10, 10), text, font=font, fill=(255, 255, 0)) 
    text = 'Ego-Vehicle Image'
    draw.text((10, H_1 + 10 + 20), text, font=font, fill=(255, 255, 0)) 
    text = 'UniV2X Planning'
    draw.text((W_1 + 10 + 20, 60), text, font=font, fill=(255, 0, 0)) 
    
    return merged_image

if __name__ == '__main__':
    import os
    folder_path = './output_visualize/1126_fusion_results'
    data_root = '/home/yangzhenwei/coding/UniV2X/datasets/V2X-Seq-SPD-Batch-95-2-3134-new-velo-lidar'
    inf_image_root = '/home/yangzhenwei/coding/UniV2X/tools/visualize/image_with_label'

    save_merged_path = './output_visualize/1126_fusion_results_merged_with_ego_label'
    if not osp.exists(save_merged_path):       
        os.makedirs(save_merged_path)

    coop_data_info_path = osp.join(data_root, 'cooperative/data_info.json')
    coop_data_infos = load_json(coop_data_info_path)
    veh_inf_mappings = {}
    for data_info in coop_data_infos:
        veh_frame_id = data_info['vehicle_frame']
        inf_frame_id = data_info['infrastructure_frame']
        veh_inf_mappings[veh_frame_id] = inf_frame_id

    merged_images_scenes = {}
    output_imgs_path = glob.glob(osp.join(folder_path, '*.jpg'))
    output_imgs_path = sorted(output_imgs_path)

    # for img_path in tqdm(output_imgs_path):
    for i in range(len(output_imgs_path)):
        img_path = output_imgs_path[i]
        sample_id = img_path.split('/')[-1].split('.')[0].split('_')[-1]
        scene_id = img_path.split('/')[-1].split('.')[0].split('_')[1]
        # if scene_id not in merged_images_scenes.keys():
        #     merged_images_scenes[scene_id] = []
        # img_inf_path = osp.join(data_root, 'infrastructure-side/image', veh_inf_mappings[sample_id] + '.jpg')
        img_inf_path = osp.join(inf_image_root, veh_inf_mappings[sample_id] + '.jpg')
        img_veh_path = osp.join(data_root, 'vehicle-side/image', sample_id + '.jpg')

        merged_image = merge_images(img_inf_path, img_veh_path, img_path)
        
        cur_save_merged_image_path = osp.join(save_merged_path, scene_id)
        if not osp.exists(cur_save_merged_image_path):       
            os.makedirs(cur_save_merged_image_path)

        save_merged_image_path = osp.join(cur_save_merged_image_path, sample_id + '.jpg')
        merged_image.save(save_merged_image_path)

        # merged_images_scenes[scene_id].append(np.array(merged_image)[:, :, ::-1])
        merged_images_scenes[scene_id] = cur_save_merged_image_path

    for scene_id in merged_images_scenes.keys():
        cur_save_merged_image_path = merged_images_scenes[scene_id]

        out_path = osp.join(save_merged_path, scene_id + '.mp4')
        height, width = merged_image.height, merged_image.width
        size = (width, height)
        fps = str(4)
        # out = cv2.VideoWriter(
        #     out_path, cv2.VideoWriter_fourcc(*('DIVX')), fps, size) #mp4v
        # for ii in range(len(merged_images_each_scene)):
        #     out.write(merged_images_each_scene[ii])
        # out.release()

        img_path = cur_save_merged_image_path + '/*.jpg' 
        commd = 'ffmpeg -framerate ' + fps + ' -pattern_type glob -i ' + "'" + img_path + "'" + '  -c:v libx264 -pix_fmt yuv420p  ' + out_path
        os.system(commd)



